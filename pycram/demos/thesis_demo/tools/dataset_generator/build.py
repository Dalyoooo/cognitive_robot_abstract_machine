import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ...planner.prompt import system_prompt, user_turn

from .catalog import DEFAULT_CATALOG_PATH, export_catalog
from .counterfactual import make_counterfactual_scenario
from .domain import Mention, RawExample
from .language import render_instruction
from .resolver import (
    clarification_question,
    natural_reference,
    resolve_mentions,
)
from .scenarios import FAMILIES, ScenarioSampler, compile_plan
from .validate import (
    validate_messages,
    validate_plan,
    validate_unique_examples,
)
from .world import (
    DEFAULT_HOLDOUT_PATH,
    HoldoutPolicy,
    WorldComposer,
)

VARIABLE_AMBIGUITY_FAMILIES = {
    "clarify_open_container",
    "clarify_destination",
}
MAX_SAMPLING_ATTEMPTS = 30
MIN_AMBIGUITY_SIZE = 2
MAX_AMBIGUITY_SIZE = 12


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _selected_value(scenario, resolution):
    if resolution.slot is None:
        raise ValueError("clarification has no unresolved slot")
    wanted = getattr(scenario.intent, resolution.slot)
    if wanted in resolution.candidates:
        return str(wanted)
    if resolution.candidates:
        raise ValueError(
            f"intended {resolution.slot} {wanted!r} is not one of "
            f"{list(resolution.candidates)!r}"
        )
    if wanted is None:
        raise ValueError(f"scenario has no value for missing slot {resolution.slot}")
    return str(wanted)


def _answer_reference(selected, resolution, scenario, catalog):
    if resolution.slot == "object":
        role = (
            "container" if resolution.intent.action in {"open", "close"} else "object"
        )
        names = scenario.context.names_for_role(role)
    elif resolution.slot == "source":
        role = "place"
        names = scenario.context.object_locations.get(
            resolution.intent.object or "", ()
        )
    else:
        role = scenario.context.role_of(selected) or "place"
        names = scenario.context.names_for_role(role)
    text = natural_reference(
        selected,
        resolution.candidates or names,
        scenario.context,
        catalog,
    )
    return text, Mention(resolution.slot, text, role)


def _resolve_scenario(scenario, rendered, catalog):
    """Resolve every rendered clause and check the scenario's expected outcome."""
    clauses = rendered.clauses or (rendered,)
    resolutions = []
    for intent, clause in zip(scenario.intents(), clauses, strict=True):
        resolution = resolve_mentions(intent, clause.mentions, scenario.context, catalog)
        resolutions.append(resolution)
    resolutions = tuple(resolutions)
    expects_clarification = scenario.family.startswith("clarify_")

    if expects_clarification and len(resolutions) > 1:
        raise ValueError("multi-task clarification is not supported")
    if expects_clarification and resolutions[0].status == "resolved":
        raise ValueError(f"{scenario.family} must require clarification")
    if not expects_clarification:
        statuses = [item.status for item in resolutions]
        if any(status != "resolved" for status in statuses):
            raise ValueError(f"{scenario.family} must resolve directly, got {statuses}")
    return resolutions, expects_clarification


def _initial_messages(rendered, context_dict):
    """Build the system and first user messages shared by every example."""
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_turn(rendered.text, context_dict)},
    ]


def _clarification_dialogue(
    scenario,
    rendered,
    resolution,
    context_dict,
    catalog,
):
    """Resolve one clarification answer and return its two dialogue turns."""
    context = scenario.context
    question = clarification_question(resolution, context, catalog)
    selected = _selected_value(scenario, resolution)
    answer_text, answer_mention = _answer_reference(
        selected,
        resolution,
        scenario,
        catalog,
    )
    remaining_mentions = tuple(
        mention for mention in rendered.mentions if mention.slot != resolution.slot
    )
    followup = resolve_mentions(
        resolution.intent,
        (*remaining_mentions, answer_mention),
        context,
        catalog,
    )
    if followup.status != "resolved":
        raise ValueError(
            "clarification answer did not fully resolve the instruction: "
            f"{followup.status} {followup.slot}"
        )

    turns = [
        {
            "role": "assistant",
            "content": _json({"clarification": question}),
        },
        {
            "role": "user",
            "content": user_turn(answer_text, context_dict),
        },
    ]
    return question, followup.intent, turns


def _compile_intents(intents, context, catalog):
    """Compile and validate intents in conversation order."""
    steps = []
    for intent in intents:
        intent_steps = compile_plan(intent, context, catalog)
        validate_plan(intent, intent_steps, context, catalog)
        steps.extend(intent_steps)
    return tuple(steps)


def build_example_from_rendered(scenario, rendered, catalog):
    """
    Build an example from an instruction that has already been rendered.

    Args:
        scenario: Scenario containing the intended task and world context.
        rendered: Rendered instruction and its entity mentions.
        catalog: Semantic catalog used to resolve references and compile plans.

    Returns:
        A pair containing the validated raw example and rendered instruction.

    Raises:
        ValueError: If grounding, plan compilation, or validation fails.
    """
    context = scenario.context
    context_dict = context.to_dict()
    resolutions, expects_clarification = _resolve_scenario(
        scenario,
        rendered,
        catalog,
    )
    messages = _initial_messages(rendered, context_dict)

    clarification = None
    if expects_clarification:
        clarification, final_intent, dialogue_turns = _clarification_dialogue(
            scenario,
            rendered,
            resolutions[0],
            context_dict,
            catalog,
        )
        final_intents = (final_intent,)
        messages.extend(dialogue_turns)
    else:
        final_intents = tuple(item.intent for item in resolutions)

    plan = _compile_intents(final_intents, context, catalog)
    messages.append(
        {
            "role": "assistant",
            "content": _json({"plan": [step.to_dict() for step in plan]}),
        }
    )
    example = RawExample(
        scenario_id=scenario.id,
        family=scenario.family,
        world_id=scenario.world_id,
        context=context,
        messages=messages,
        plan=plan,
        clarification=clarification,
        template_id=rendered.template_id,
        metadata={
            "resolution": "+".join(resolution.status for resolution in resolutions)
        },
    )
    validate_messages(example)
    return example, rendered


def build_example(scenario, catalog, rng):
    """
    Render and build one direct plan or clarification dialogue.

    Args:
        scenario: Scenario to render and compile.
        catalog: Semantic catalog used for grounding and plan compilation.
        rng: Random generator used to select a language template.

    Returns:
        A pair containing the validated raw example and rendered instruction.

    Raises:
        ValueError: If rendering, grounding, compilation, or validation fails.
    """
    rendered = render_instruction(scenario, rng)
    return build_example_from_rendered(scenario, rendered, catalog)


def _compose_context(composer, family, family_index, rng):
    """Compose a normal world or a larger ambiguity-training world."""
    if family not in VARIABLE_AMBIGUITY_FAMILIES:
        return composer.compose(rng)
    number_of_sizes = MAX_AMBIGUITY_SIZE - MIN_AMBIGUITY_SIZE + 1
    duplicate_count = MIN_AMBIGUITY_SIZE + family_index % number_of_sizes
    return composer.compose(rng, duplicate_count=duplicate_count)


def _generate_example(composer, sampler, catalog, family, family_index, serial, seed):
    """Generate one example, retrying only world/scenario sampling failures."""
    last_error = None
    for attempt in range(MAX_SAMPLING_ATTEMPTS):
        rng = random.Random(f"{seed}:{family}:{family_index}:{attempt}")
        try:
            context = _compose_context(composer, family, family_index, rng)
            scenario = sampler.sample(
                family,
                context,
                rng,
                serial,
                world_id=f"composed-{serial}",
            )
        except ValueError as error:
            last_error = error
            continue

        # Contract errors from rendering, grounding, or validation must fail
        # immediately instead of being hidden by another sampling attempt.
        example, rendered = build_example(scenario, catalog, rng)
        return scenario, example, rendered

    raise RuntimeError(
        f"could not generate {family}[{family_index}]: {last_error}"
    ) from last_error


def _context_pair(scenario, example, rendered, catalog):
    """Return a base example and, when possible, its context counterfactual."""
    counterfactual_scenario = make_counterfactual_scenario(scenario)
    if counterfactual_scenario is None:
        return ((scenario, example, rendered),)

    counterfactual, _ = build_example_from_rendered(
        counterfactual_scenario,
        rendered,
        catalog,
    )
    if counterfactual.plan == example.plan:
        raise ValueError("counterfactual context did not change the target plan")

    pair_id = scenario.id
    example.metadata.update({"pair_id": pair_id, "context_variant": "base"})
    counterfactual.metadata.update(
        {"pair_id": pair_id, "context_variant": "counterfactual"}
    )
    return (
        (scenario, example, rendered),
        (counterfactual_scenario, counterfactual, rendered),
    )


def generate_base_examples(composer, sampler, catalog, *, per_family, seed):
    """
    Generate a fixed quota from every scenario family.

    Args:
        composer: World composer used to create symbolic contexts.
        sampler: Scenario sampler used to select canonical tasks.
        catalog: Semantic catalog used to build and validate examples.
        per_family: Number of base scenarios to generate per family.
        seed: Seed used to derive deterministic per-scenario generators.

    Returns:
        Three parallel lists containing scenarios, examples, and rendered rows.

    Raises:
        RuntimeError: If a valid world and scenario cannot be sampled.
        ValueError: If a generated example violates a build contract.
    """
    work = []
    for family in FAMILIES:
        for family_index in range(per_family):
            work.append((family, family_index))
    scenarios = []
    examples = []
    rendered_rows = []
    for serial, (family, family_index) in enumerate(work):
        scenario, example, rendered = _generate_example(
            composer,
            sampler,
            catalog,
            family,
            family_index,
            serial,
            seed,
        )
        for paired_scenario, paired_example, paired_rendered in _context_pair(
            scenario,
            example,
            rendered,
            catalog,
        ):
            scenarios.append(paired_scenario)
            examples.append(paired_example)
            rendered_rows.append(paired_rendered)
    return scenarios, examples, rendered_rows


def split_examples(examples, seed):
    """
    Split examples by task family while keeping context pairs together.

    Args:
        examples: Validated examples to divide into train and validation sets.
        seed: Seed used to shuffle family groups and final splits.

    Returns:
        A pair containing the training and validation example lists.
    """
    by_family = {}
    for example in examples:
        group_id = example.metadata.get("pair_id", example.scenario_id)
        family_groups = by_family.setdefault(example.family, {})
        family_groups.setdefault(group_id, []).append(example)

    train = []
    val = []
    rng = random.Random(seed)
    for _family, family_groups in sorted(by_family.items()):
        groups = list(family_groups.values())
        rng.shuffle(groups)
        val_count = max(1, round(len(groups) * 0.1)) if len(groups) > 1 else 0
        for group in groups[:val_count]:
            val.extend(group)
        for group in groups[val_count:]:
            train.extend(group)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _training_row(example, split):
    template_family = example.metadata.get("pair_id", example.scenario_id)
    return {
        "messages": example.messages,
        "metadata": {
            "scenario_id": example.scenario_id,
            "family": example.family,
            "task_type": example.family,
            "world_id": example.world_id,
            "template_id": example.template_id,
            "template_family": template_family,
            "split": split,
            "context_dict": example.context.to_dict(),
            **example.metadata,
        },
    }


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _scenario_row(scenario):
    return {
        "id": scenario.id,
        "family": scenario.family,
        "world_id": scenario.world_id,
        "intents": [asdict(intent) for intent in scenario.intents()],
        "reference_sets": list(scenario.reference_sets()),
        "context": scenario.context.to_dict(),
    }


def write_dataset(
    output_dir,
    scenarios,
    train,
    val,
    *,
    seed,
    per_family,
):
    """
    Write dataset rows and build metadata to an output directory.

    Args:
        output_dir: Directory that receives the generated artifacts.
        scenarios: Generated scenarios to write for inspection.
        train: Examples assigned to the training split.
        val: Examples assigned to the validation split.
        seed: Seed recorded in the manifest.
        per_family: Base scenario quota recorded in the manifest.

    Raises:
        OSError: If an output directory or file cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_examples = train + val
    _write_jsonl(
        output_dir / "train.jsonl",
        (_training_row(row, "train") for row in train),
    )
    _write_jsonl(
        output_dir / "val.jsonl",
        (_training_row(row, "val") for row in val),
    )
    _write_jsonl(output_dir / "scenarios.jsonl", map(_scenario_row, scenarios))

    review = []
    for family in FAMILIES:
        row = None
        for item in all_examples:
            if item.family == family:
                row = item
                break
        if row is not None:
            review.append(_training_row(row, "review"))
    _write_jsonl(output_dir / "review.jsonl", review)

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "per_family": per_family,
        "counts": {
            "train": len(train),
            "val": len(val),
            "families": dict(Counter(row.family for row in all_examples)),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_holdouts(examples, policy, catalog):
    """
    Reject examples that contain held-out entities or terms.

    Args:
        examples: Examples whose contexts and messages should be checked.
        policy: Holdout policy defining forbidden entities and terms.
        catalog: Semantic catalog used to expand forbidden type labels.

    Raises:
        ValueError: If any context or message contains held-out data.
    """
    for example in examples:
        policy.assert_clean(example.context, catalog)
        policy.assert_text_clean(json.dumps(example.messages), catalog)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("planner_data"))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT_PATH)
    return parser.parse_args()


def main():
    """
    Generate the dataset from command-line arguments.

    Raises:
        SystemExit: If the requested per-family quota is not positive.
    """
    args = _parse_args()
    if args.per_family < 1:
        raise SystemExit("--per-family must be positive")
    # Exporting is cheap and prevents a committed catalog from silently
    # becoming stale after semDT or the small planner policy changes.
    catalog = export_catalog(args.catalog)
    holdout = HoldoutPolicy.load(args.holdout)
    composer = WorldComposer(catalog, holdout)
    sampler = ScenarioSampler(catalog)
    scenarios, examples, _ = generate_base_examples(
        composer,
        sampler,
        catalog,
        per_family=args.per_family,
        seed=args.seed,
    )

    validate_holdouts(examples, holdout, catalog)
    validate_unique_examples(examples)
    train, val = split_examples(examples, args.seed)
    write_dataset(
        args.out_dir,
        scenarios,
        train,
        val,
        seed=args.seed,
        per_family=args.per_family,
    )
    print(f"Wrote {len(train)} train and {len(val)} validation rows to {args.out_dir}")


if __name__ == "__main__":
    main()
