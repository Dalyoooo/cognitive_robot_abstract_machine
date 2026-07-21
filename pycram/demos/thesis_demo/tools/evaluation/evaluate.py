import argparse
import csv
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import llama_cpp

ROOT = Path(
    os.environ.get(
        "NLP_DATASET_ROOT",
        Path(__file__).resolve().parents[5].parent / "NLP-binder",
    )
)

from ...planner import llm as planner

from .demo_session import DemoSession
from .pipeline import (
    create_log,
    evaluate_case,
    log_event,
    log_exception,
    validate_demo_world,
)
from .report import generate_report
from .scoring import (
    CASES_FILE,
    EvaluationConfig,
    LiveSummary,
    load_cases,
)


def write_outputs(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "results.json",
        output_dir / "results.csv",
    )
    report = {
        "results": [asdict(result) for result in results],
    }
    paths[0].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = [result.to_row() for result in results]
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with paths[1].open("w", newline="", encoding="utf-8") as handle:
        if columns:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return paths


def _parser():
    parser = argparse.ArgumentParser(
        description="Run the frozen Kitchen benchmark through planning and CRAM execution."
    )
    parser.add_argument("--model")
    parser.add_argument("--gguf-file")
    parser.add_argument("--cases", type=Path, default=CASES_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/kitchen"))
    parser.add_argument("--validate-world", action="store_true")
    parser.add_argument("--world-timeout", type=float, default=60.0)
    parser.add_argument("--execution-timeout", type=float, default=180.0)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only this case ID. May be repeated.",
    )
    parser.add_argument(
        "--limit", type=int, help="Run only the first N selected cases."
    )
    parser.add_argument(
        "--visualization-delay",
        type=float,
        default=0.0,
        help="Pause before execution and between cases for RViz viewing.",
    )
    parser.add_argument("--n-gpu-layers", type=int)
    parser.add_argument("--n-ctx", type=int, default=8192)
    return parser


def _select_cases(parser, args):
    cases = load_cases(args.cases)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.id in selected]
        missing = selected - {case.id for case in cases}
        if missing:
            parser.error(f"unknown --case-id values: {', '.join(sorted(missing))}")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        cases = cases[: args.limit]
    return cases


def _format_generated_plan(result):
    if result.plan is None:
        return "no plan"
    return json.dumps(result.plan, sort_keys=True, default=str)


def _configure_runtime_dir(path):
    runtime_dir = Path(path).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NLP_RUN_DIR"] = str(runtime_dir)
    return runtime_dir


def _run_cases(args, cases, planner, session, config, logger):
    results = []
    paths = ()
    for index, case in enumerate(cases):
        log_event(
            logger,
            "case_start",
            {
                "case_id": case.id,
                "phase": "case_setup",
                "expected_outcome": case.expected_outcome,
            },
        )
        result = evaluate_case(case, planner, session, config, logger)
        log_event(
            logger,
            "plan_generated",
            {
                "case_id": case.id,
                "phase": "planning",
                "planner_outcome": result.planner_outcome,
                "plan": result.plan,
            },
        )
        results.append(result)
        paths = write_outputs(results, args.output_dir)
        log_event(
            logger,
            "artifacts_written",
            {
                "case_id": case.id,
                "phase": "artifacts",
                "paths": [str(path) for path in paths],
            },
        )
        log_event(
            logger,
            "case_end",
            {
                "case_id": case.id,
                "phase": result.failure_stage or "complete",
                "success": result.task_success,
                "execution_status": result.execution_status,
                "failure_stage": result.failure_stage,
                "error": result.error,
                "total_latency_s": result.total_latency_s,
            },
            logging.INFO if result.task_success else logging.ERROR,
        )
        status = "PASS" if result.task_success else "FAIL"
        print(f"[{status}] {case.id}: {result.execution_status}")
        print(f"  generated plan: {_format_generated_plan(result)}")
        if result.error:
            print(f"  {result.error}")
        if args.visualization_delay > 0 and index < len(cases) - 1:
            time.sleep(args.visualization_delay)
    return results, paths


def main():
    parser = _parser()
    args = parser.parse_args()
    if not args.validate_world and not args.model:
        parser.error("--model is required unless --validate-world is used")
    cases = _select_cases(parser, args)
    if not cases:
        parser.error("no evaluation cases selected")
    config = EvaluationConfig(
        world_timeout_s=args.world_timeout,
        execution_timeout_s=args.execution_timeout,
        visualization_delay_s=args.visualization_delay,
    )
    session = DemoSession(visualize=os.environ.get("NLP_VISUALIZE", "1") != "0")

    if args.validate_world:
        with tempfile.TemporaryDirectory(prefix="nlp-binder-validate-") as run_dir:
            _configure_runtime_dir(run_dir)
            validate_demo_world(cases, session, config)
        print(f"Validated {len(cases)} cases against the live CRAM Kitchen.")
        return 0

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = _configure_runtime_dir(args.output_dir / ".runtime")

    log_path = args.output_dir / "evaluation.log"
    logger = create_log(log_path)
    log_event(
        logger,
        "run_start",
        {
            "case_id": None,
            "phase": "run_setup",
            "model": args.model,
            "gguf_file": args.gguf_file,
            "n_ctx": args.n_ctx,
            "max_tokens": planner.MAX_NEW_TOKENS,
            "llama_cpp_version": llama_cpp.__version__,
            "cases": len(cases),
            "case_file": str(args.cases),
            "runtime_dir": str(runtime_dir),
        },
    )
    gguf_file = args.gguf_file or (args.model if Path(args.model).is_file() else None)
    run_phase = "planner_setup"
    try:
        planner.setup_planner(
            args.model,
            gguf_file=gguf_file,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
        )
        run_phase = "cases"
        results, paths = _run_cases(args, cases, planner, session, config, logger)
        summary = LiveSummary.from_results(results)
        run_phase = "report"
        report_paths = generate_report(paths[1], args.output_dir)
    except Exception as exc:
        log_exception(
            logger,
            "run_failure",
            {
                "case_id": None,
                "phase": run_phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output_dir": str(args.output_dir),
            },
        )
        raise
    log_event(
        logger,
        "run_end",
        {
            "case_id": None,
            "phase": "complete",
            **asdict(summary),
            "output_dir": str(args.output_dir),
        },
    )
    planner_rate = summary.metric_rates["planner_success"]["percent"]
    task_rate = summary.metric_rates["task_success"]["percent"]
    planner_text = f"{planner_rate:.1f}%" if planner_rate is not None else "n/a"
    task_text = f"{task_rate:.1f}%" if task_rate is not None else "n/a"
    print(f"Planner: {planner_text}, task success: {task_text}")
    print("Wrote " + ", ".join(map(str, (*paths, log_path, *report_paths))))
    return 0 if task_rate == 100.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
