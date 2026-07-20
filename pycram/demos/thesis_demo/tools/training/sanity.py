import json

from ...planner.prompt import system_prompt, user_turn
from ...validation.guard import verify


class SanityCheckError(RuntimeError):
    """Expected failure of a learned-behavior check."""


def generate(model, tokenizer, messages, max_new_tokens=2048):
    """Generate one deterministic assistant response with an attention mask."""
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    attention_mask = tokens.new_ones(tokens.shape)
    output = model.generate(
        tokens,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(
        output[0][tokens.shape[1] :], skip_special_tokens=True
    ).strip()


def run_sanity_check(model, tokenizer, generate_response, fast_model):
    """Verify container and surface transport behavior."""
    fast_model.for_inference(model)
    cases = [
        {
            "name": "Fridge source",
            "instruction": "bring the milk from the fridge to the table",
            "context": {
                "objects": ["milk"],
                "object_locations": {"milk": ["fridge"]},
                "surfaces": ["table"],
                "containers": ["fridge"],
                "openables": ["fridge"],
                "furniture": [],
                "rooms": ["kitchen"],
                "types": {
                    "milk": "Milk",
                    "table": "Table",
                    "fridge": "Fridge",
                    "kitchen": "Kitchen",
                },
            },
            "container_actions": {"OpenAction", "CloseAction"},
            "source": "fridge",
            "destination": "table",
        },
        {
            "name": "Fridge source rephrase",
            "instruction": "fetch the milk from the fridge and place it on the table",
            "context": {
                "objects": ["milk"],
                "object_locations": {"milk": ["fridge"]},
                "surfaces": ["table"],
                "containers": ["fridge"],
                "openables": ["fridge"],
                "furniture": [],
                "rooms": ["kitchen"],
                "types": {
                    "milk": "Milk",
                    "table": "Table",
                    "fridge": "Fridge",
                    "kitchen": "Kitchen",
                },
            },
            "container_actions": {"OpenAction", "CloseAction"},
            "source": "fridge",
            "destination": "table",
        },
        {
            "name": "surface transport",
            "instruction": "move the milk from the counter to the table",
            "context": {
                "objects": ["milk"],
                "object_locations": {"milk": ["counter"]},
                "surfaces": ["counter", "table"],
                "containers": [],
                "openables": [],
                "furniture": [],
                "rooms": ["kitchen"],
                "types": {
                    "milk": "Milk",
                    "counter": "CounterTop",
                    "table": "Table",
                    "kitchen": "Kitchen",
                },
            },
            "container_actions": set(),
            "source": "counter",
            "destination": "table",
        },
    ]

    for case in cases:
        messages = [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": user_turn(case["instruction"], case["context"]),
            },
        ]
        text = generate_response(model, tokenizer, messages)
        try:
            plan = json.loads(text)
        except json.JSONDecodeError as error:
            raise SanityCheckError(
                f"{case['name']} failed: response is not JSON: {text[:200]!r}."
            ) from error
        valid, reason = verify(plan, case["context"])
        if not valid or not plan.get("plan"):
            raise SanityCheckError(f"{case['name']} failed: invalid plan: {reason}.")
        transports = [
            step for step in plan["plan"] if step.get("action") == "TransportAction"
        ]
        if len(transports) != 1 or (
            transports[0].get("object"),
            transports[0].get("source"),
            transports[0].get("location"),
        ) != ("milk", case["source"], case["destination"]):
            raise SanityCheckError(
                f"{case['name']} failed: expected one matching TransportAction."
            )
        container_actions = {
            step["action"]
            for step in plan["plan"]
            if step["action"] in {"OpenAction", "CloseAction"}
        }
        if container_actions != case["container_actions"]:
            raise SanityCheckError(
                f"{case['name']} failed: expected container actions "
                f"{sorted(case['container_actions'])}, got "
                f"{sorted(container_actions)}."
            )
    print(">>> Sanity check passed: all 3 planner behavior cases are valid.")
