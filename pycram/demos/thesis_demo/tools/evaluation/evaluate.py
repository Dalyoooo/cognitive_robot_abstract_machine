from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import llama_cpp

from thesis_demo.planner import llm as planner
from thesis_demo.tools.evaluation.demo_session import DemoSession
from thesis_demo.tools.evaluation.pipeline import (
    create_log,
    evaluate_case,
    log_event,
    log_exception,
    validate_demo_world,
)
from thesis_demo.tools.evaluation.scoring import (
    CASES_FILE,
    MAIN_METRICS,
    EvaluationConfiguration,
    EvaluationMode,
    LiveSummary,
    MetricAggregate,
    ModelVariant,
    load_cases,
)


@dataclass(frozen=True)
class ModelSpecification:
    variant: ModelVariant
    model: str
    gguf_file: str | None


class EvaluationStoppedAfterTimeout(RuntimeError):
    pass


@dataclass
class ResultWriter:
    output_directory: Path
    results: list = field(default_factory=list)

    @property
    def results_path(self):
        return self.output_directory / "results.jsonl"

    @property
    def case_results_path(self):
        return self.output_directory / "results.csv"

    @property
    def summary_path(self):
        return self.output_directory / "summary.csv"

    def append(self, result):
        self.results.append(result)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_record(), sort_keys=True) + "\n")
        self.write_case_results(self.results)
        self.write_summary(self.results)

    def write_case_results(self, results):
        rows = [
            {
                "id": result.case.id,
                "model_variant": (
                    result.model_variant.value
                    if result.model_variant is not None
                    else None
                ),
                "evaluation_mode": result.evaluation_mode.value,
                "expected_outcome": result.case.expected_outcome,
                "planner_outcome": result.planner_outcome,
                "planning_success": result.planning_success,
                "grounding_success": result.grounding_success,
                "execution_success": result.execution_success,
                "task_success": result.task_success,
                "execution_status": result.execution_status,
                "timed_out": result.timed_out,
                "failure_stage": result.failure_stage,
                "error": result.error,
                "planning_latency_s": result.planning_latency_s,
                "execution_latency_s": result.execution_latency_s,
                "total_latency_s": result.total_latency_s,
            }
            for result in results
        ]
        self._write_csv(self.case_results_path, rows)
        return self.case_results_path

    def write_summary(self, results):
        grouped = {
            variant: [result for result in results if result.model_variant == variant]
            for variant in ModelVariant
        }
        rows = []
        for metric in MAIN_METRICS:
            base = MetricAggregate.from_results(grouped[ModelVariant.BASE], metric)
            finetuned = MetricAggregate.from_results(
                grouped[ModelVariant.FINETUNED], metric
            )
            rows.append(
                {
                    "metric": metric,
                    "base_successes": base.successes,
                    "base_observed_cases": base.observed_cases,
                    "base_total_cases": base.total_cases,
                    "base_stage_percent": base.stage_percent,
                    "base_benchmark_percent": base.benchmark_percent,
                    "finetuned_successes": finetuned.successes,
                    "finetuned_observed_cases": finetuned.observed_cases,
                    "finetuned_total_cases": finetuned.total_cases,
                    "finetuned_stage_percent": finetuned.stage_percent,
                    "finetuned_benchmark_percent": finetuned.benchmark_percent,
                    "stage_difference_percentage_points": _difference(
                        base.stage_percent, finetuned.stage_percent
                    ),
                    "benchmark_difference_percentage_points": _difference(
                        base.benchmark_percent, finetuned.benchmark_percent
                    ),
                }
            )

        self._write_csv(self.summary_path, rows)
        return self.summary_path

    @staticmethod
    def _write_csv(path, rows):
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)


def _difference(base_percent, finetuned_percent):
    if base_percent is None or finetuned_percent is None:
        return None
    return round(finetuned_percent - base_percent, 1)


def _positive_seconds(value):
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return seconds


def _nonnegative_seconds(value):
    seconds = float(value)
    if seconds < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return seconds


def create_parser():
    argument_parser = argparse.ArgumentParser(
        description=(
            "Compare a base and fine-tuned planner on the frozen Kitchen benchmark."
        )
    )
    argument_parser.add_argument("--base-model")
    argument_parser.add_argument("--base-gguf-file")
    argument_parser.add_argument("--finetuned-model")
    argument_parser.add_argument("--finetuned-gguf-file")
    argument_parser.add_argument("--cases", type=Path, default=CASES_FILE)
    argument_parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_results/kitchen")
    )
    argument_parser.add_argument("--validate-world", action="store_true")
    argument_parser.add_argument(
        "--mode",
        type=EvaluationMode,
        choices=tuple(EvaluationMode),
        default=EvaluationMode.END_TO_END,
    )
    argument_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only this case ID. May be repeated.",
    )
    argument_parser.add_argument("--visualization-delay", type=float, default=0.0)
    argument_parser.add_argument(
        "--world-settle-delay",
        type=_nonnegative_seconds,
        default=2.0,
    )
    argument_parser.add_argument(
        "--case-timeout",
        type=_positive_seconds,
        default=EvaluationConfiguration().case_timeout_s,
        help=(
            "Maximum seconds without completed progress. Each completed plan "
            "step resets the timer."
        ),
    )
    argument_parser.add_argument("--n-gpu-layers", type=int)
    argument_parser.add_argument("--n-ctx", type=int, default=planner.N_CTX)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--temperature", type=float, default=0.0)
    argument_parser.add_argument(
        "--max-tokens", type=int, default=planner.MAX_NEW_TOKENS
    )
    return argument_parser


def select_cases(argument_parser, arguments):
    cases = load_cases(arguments.cases)
    if not arguments.case_id:
        return cases

    selected_ids = set(arguments.case_id)
    selected_cases = [case for case in cases if case.id in selected_ids]
    missing_ids = selected_ids - {case.id for case in selected_cases}
    if missing_ids:
        argument_parser.error(
            f"unknown --case-id values: {', '.join(sorted(missing_ids))}"
        )
    return selected_cases


def model_specifications(argument_parser, arguments):
    missing = [
        option
        for option, value in (
            ("--base-model", arguments.base_model),
            ("--finetuned-model", arguments.finetuned_model),
        )
        if not value
    ]
    if missing:
        argument_parser.error(f"required arguments: {', '.join(missing)}")
    models = (
        ModelSpecification(
            ModelVariant.BASE,
            arguments.base_model,
            _gguf_file(arguments.base_model, arguments.base_gguf_file),
        ),
        ModelSpecification(
            ModelVariant.FINETUNED,
            arguments.finetuned_model,
            _gguf_file(arguments.finetuned_model, arguments.finetuned_gguf_file),
        ),
    )
    missing_gguf = [
        option
        for option, model in (
            ("--base-gguf-file", models[0]),
            ("--finetuned-gguf-file", models[1]),
        )
        if model.gguf_file is None
    ]
    if missing_gguf:
        argument_parser.error(
            "GGUF filenames are required for repository models: "
            + ", ".join(missing_gguf)
        )
    return models


def _gguf_file(model, configured_file):
    if configured_file:
        return configured_file
    if Path(model).is_file():
        return model
    return None


def _run_cases(cases, model, session, configuration, inference, logger, writer):
    results = []
    for case_index, case in enumerate(cases):
        case_inference = inference.for_case(case.id)
        log_event(
            logger,
            "case_start",
            {
                "case_id": case.id,
                "model_variant": model.variant.value,
                "evaluation_mode": configuration.mode.value,
                "inference": asdict(case_inference),
            },
        )
        result = evaluate_case(
            case,
            planner,
            session,
            configuration,
            logger,
            model_variant=model.variant,
            inference=case_inference,
        )
        results.append(result)
        writer.append(result)
        _print_result(result)
        log_event(logger, "case_end", result.to_record())
        if result.timed_out:
            raise EvaluationStoppedAfterTimeout(
                f"case {case.id} timed out. The native pyCRAM state may be "
                "inconsistent. Preserve these results and restart the evaluation "
                "in a fresh process."
            )
        if configuration.visualization_delay_s > 0 and case_index < len(cases) - 1:
            time.sleep(configuration.visualization_delay_s)
    return results


def _print_result(result):
    success = (
        result.planning_success
        if result.evaluation_mode == EvaluationMode.PLANNING
        else result.task_success
    )
    status = "PASS" if success else "FAIL"
    print(
        f"[{status}] {result.model_variant.value} {result.case.id}: "
        f"{result.execution_status}"
    )
    if result.error:
        print(f"  {result.error}")


def _prepare_output_directory(argument_parser, output_directory):
    if output_directory.exists() and any(output_directory.iterdir()):
        argument_parser.error(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)


def main():
    argument_parser = create_parser()
    arguments = argument_parser.parse_args()
    cases = select_cases(argument_parser, arguments)
    if not cases:
        argument_parser.error("no evaluation cases selected")

    configuration = EvaluationConfiguration(
        mode=arguments.mode,
        visualization_delay_s=arguments.visualization_delay,
        world_settle_delay_s=arguments.world_settle_delay,
        case_timeout_s=arguments.case_timeout,
    )
    session = DemoSession(visualize=os.environ.get("NLP_VISUALIZE", "1") != "0")
    if arguments.validate_world:
        validate_demo_world(cases, session, configuration.world_settle_delay_s)
        print(f"Validated {len(cases)} cases against the live CRAM Kitchen.")
        return 0

    models = model_specifications(argument_parser, arguments)
    _prepare_output_directory(argument_parser, arguments.output_dir)
    writer = ResultWriter(arguments.output_dir)
    logger = create_log(arguments.output_dir / "evaluation.log")
    inference = planner.InferenceConfiguration(
        seed=arguments.seed,
        temperature=arguments.temperature,
        max_tokens=arguments.max_tokens,
    )
    log_event(
        logger,
        "run_start",
        {
            "case_ids": [case.id for case in cases],
            "evaluation_mode": configuration.mode.value,
            "case_timeout_s": configuration.case_timeout_s,
            "world_settle_delay_s": configuration.world_settle_delay_s,
            "inference": asdict(inference),
            "n_ctx": arguments.n_ctx,
            "llama_cpp_version": llama_cpp.__version__,
            "models": [asdict(model) for model in models],
        },
    )

    results = []
    run_phase = "planner_setup"
    try:
        for model in models:
            planner.setup_planner(
                model.model,
                gguf_file=model.gguf_file,
                n_gpu_layers=arguments.n_gpu_layers,
                n_ctx=arguments.n_ctx,
                inference=inference,
            )
            run_phase = f"{model.variant.value}_cases"
            results.extend(
                _run_cases(
                    cases, model, session, configuration, inference, logger, writer
                )
            )
    except EvaluationStoppedAfterTimeout as error:
        log_event(
            logger,
            "run_stopped_after_timeout",
            {
                "phase": run_phase,
                "error": str(error),
            },
        )
        print(f"[STOP] {error}", flush=True)
        return 2
    except Exception as error:
        log_exception(
            logger,
            "run_failure",
            {
                "phase": run_phase,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    summary_path = writer.summary_path
    summaries = {
        variant.value: asdict(
            LiveSummary.from_results(
                [result for result in results if result.model_variant == variant]
            )
        )
        for variant in ModelVariant
    }
    log_event(logger, "run_end", {"summaries": summaries})
    print(
        f"Wrote {writer.results_path}, {writer.case_results_path}, "
        f"{summary_path}, and evaluation.log"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
