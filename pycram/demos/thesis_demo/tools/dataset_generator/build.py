import argparse
import json
import random
from collections import Counter
from dataclasses import replace
from enum import Enum, auto
from pathlib import Path

from thesis_demo.planner.prompt import system_prompt, user_turn
from thesis_demo.tools.dataset_generator.catalog import (
    DEFAULT_CATALOG_PATH,
    SCHEMA_VERSION,
    SemanticCatalog,
)
from thesis_demo.tools.dataset_generator.domain import Mention, RawExample, WorldNames
from thesis_demo.tools.dataset_generator.language import render_instruction
from thesis_demo.tools.dataset_generator.resolver import (
    clarification_question,
    natural_reference,
    resolve_mentions,
)
from thesis_demo.tools.dataset_generator.scenarios import (
    FAMILIES,
    ScenarioSampler,
    compile_plan,
)
from thesis_demo.tools.dataset_generator.validate import (
    validate_messages,
    validate_plan,
    validate_unique_examples,
)
from thesis_demo.tools.dataset_generator.world import (
    DEFAULT_HOLDOUT_PATH,
    HoldoutPolicy,
    WorldComposer,
)

VARIABLE_AMBIGUITY_FAMILIES = frozenset(
    {"clarify_open_container", "clarify_destination"}
)
MAX_SAMPLING_ATTEMPTS = 30
MIN_AMBIGUITY_SIZE = 2
MAX_AMBIGUITY_SIZE = 12


class NameStyle(Enum):
    NATURAL = auto()
    OPAQUE = auto()

    def world_names(self, context, seed):
        if self is NameStyle.NATURAL:
            return WorldNames.natural(context)
        return WorldNames.opaque(context, seed)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _selected_value(scenario, resolution):
    if resolution.slot is None:
        raise ValueError("clarification has no unresolved slot")
    wanted = scenario.intent.value_for(resolution.slot)
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
    clauses = rendered.clauses or (rendered,)
    resolutions = []
    for intent, clause in zip(scenario.intents(), clauses, strict=True):
        resolution = resolve_mentions(
            intent, clause.mentions, scenario.context, catalog
        )
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
    steps = []
    for intent in intents:
        intent_steps = compile_plan(intent, context, catalog)
        validate_plan(intent, intent_steps, context, catalog)
        steps.extend(intent_steps)
    return tuple(steps)


def build_example_from_rendered(
    scenario,
    rendered,
    catalog,
    name_style=NameStyle.NATURAL,
):
    natural_context = scenario.context
    world_names = name_style.world_names(
        natural_context,
        seed=f"{scenario.id}:world-names",
    )
    context = world_names.context_for(natural_context)
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

    natural_plan = _compile_intents(final_intents, natural_context, catalog)
    plan = world_names.plan_for(natural_plan)
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
            "resolution": "+".join(resolution.status for resolution in resolutions),
            "name_style": name_style.name.casefold(),
        },
    )
    validate_messages(example, natural_context)
    return example, rendered


def build_example(scenario, catalog, rng):
    rendered = render_instruction(scenario, rng)
    return build_example_from_rendered(scenario, rendered, catalog)


def _compose_context(composer, family, family_index, rng):
    if family not in VARIABLE_AMBIGUITY_FAMILIES:
        return composer.compose(rng)
    number_of_sizes = MAX_AMBIGUITY_SIZE - MIN_AMBIGUITY_SIZE + 1
    duplicate_count = MIN_AMBIGUITY_SIZE + family_index % number_of_sizes
    return composer.compose(rng, duplicate_count=duplicate_count)


def _generate_example(composer, sampler, catalog, family, family_index, serial, seed):
    last_error = None
    for attempt in range(MAX_SAMPLING_ATTEMPTS):
        rng = random.Random(f"{seed}:{family}:{family_index}:{attempt}")
        try:
            # Random candidates may violate semantic constraints. Retrying with
            # the attempt seed keeps rejection deterministic and reproducible.
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

        example, rendered = build_example(scenario, catalog, rng)
        return scenario, example, rendered

    raise RuntimeError(
        f"could not generate {family}[{family_index}]: {last_error}"
    ) from last_error


def _name_pair(scenario, natural_example, rendered, catalog):
    opaque_example, _ = build_example_from_rendered(
        scenario,
        rendered,
        catalog,
        NameStyle.OPAQUE,
    )
    if opaque_example.plan == natural_example.plan:
        raise ValueError("opaque names did not change the grounded plan")

    pair_name = scenario.id
    opaque_scenario = replace(
        scenario,
        id=f"{scenario.id}-opaque",
        world_id=f"{scenario.world_id}-opaque",
    )
    opaque_example = replace(
        opaque_example,
        scenario_id=opaque_scenario.id,
        world_id=opaque_scenario.world_id,
    )
    natural_example.metadata.update({"pair_name": pair_name})
    opaque_example.metadata.update({"pair_name": pair_name})
    return (
        (scenario, natural_example, rendered),
        (opaque_scenario, opaque_example, rendered),
    )


def generate_base_examples(composer, sampler, catalog, *, per_family, seed):
    scenarios = []
    examples = []
    rendered_rows = []
    work = (
        (family, family_index)
        for family in FAMILIES
        for family_index in range(per_family)
    )
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
        for paired_scenario, paired_example, paired_rendered in _name_pair(
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
    by_family = {}
    for example in examples:
        group_name = example.metadata.get("pair_name", example.scenario_id)
        family_groups = by_family.setdefault(example.family, {})
        family_groups.setdefault(group_name, []).append(example)

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


def _training_row(example):
    return {"messages": example.messages}


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_dataset(
    output_dir,
    train,
    val,
    *,
    seed,
    per_family,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_examples = train + val
    _write_jsonl(
        output_dir / "train.jsonl",
        (_training_row(row) for row in train),
    )
    _write_jsonl(
        output_dir / "val.jsonl",
        (_training_row(row) for row in val),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
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
    args = _parse_args()
    if args.per_family < 1:
        raise SystemExit("--per-family must be positive")
    catalog = SemanticCatalog.load(args.catalog)
    holdout = HoldoutPolicy.load(args.holdout)
    composer = WorldComposer(catalog, holdout)
    sampler = ScenarioSampler(catalog)
    _scenarios, examples, _rendered = generate_base_examples(
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
        train,
        val,
        seed=args.seed,
        per_family=args.per_family,
    )
    print(f"Wrote {len(train)} train and {len(val)} validation rows to {args.out_dir}")


if __name__ == "__main__":
    main()
