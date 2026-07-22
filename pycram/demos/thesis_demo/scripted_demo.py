import argparse
import json

from .execution.execution import run_plan
from .planner.world_context import classify_world
from .world.nlp_demo import build_world

KITCHEN_PLAN = [
    {
        "action": "TransportAction",
        "object": "cheeze_it",
        "location": "table",
        "relation": "on",
        "source": "counter_top_3",
    },
    {
        "action": "ParkArmsAction",
        "object": None,
        "location": None,
        "relation": None,
        "source": None,
    },
]

APARTMENT_PLAN = [
    {
        "action": "TransportAction",
        "object": "milk",
        "location": "table",
        "relation": "on",
        "source": "counter_top_2",
    },
    {
        "action": "ParkArmsAction",
        "object": None,
        "location": None,
        "relation": None,
        "source": None,
    },
]

PLANS = {"kitchen": KITCHEN_PLAN, "apartment": APARTMENT_PLAN}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a hand-written plan in the thesis demo world."
    )
    parser.add_argument("--robot", default="hsrb")
    parser.add_argument("--environment", default="kitchen", choices=sorted(PLANS))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--print-context",
        action="store_true",
        help="Print the planner world context instead of executing the plan.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    world, robot, context, _node = build_world(
        robot_name=arguments.robot,
        environment=arguments.environment,
        visualize=arguments.visualize,
    )

    if arguments.print_context:
        print(json.dumps(classify_world(world, robot), indent=2, sort_keys=True))
        return

    steps = PLANS[arguments.environment]
    print(f"[scripted_demo] running {len(steps)} steps in {arguments.environment}")
    observations = run_plan(world, robot, context, steps)
    print(json.dumps(observations, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
