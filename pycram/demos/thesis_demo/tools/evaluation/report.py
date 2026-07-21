import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .scoring import MAIN_METRICS as SCORING_MAIN_METRICS
from .scoring import confusion_rates

METRIC_PRESENTATION = {
    "json_valid": ("Exact JSON response", "all planned cases"),
    "schema_valid": ("Schema valid", "all planned cases"),
    "guard_valid": ("Guard valid", "all planned cases"),
    "planner_success": ("Planner response success", "all planned cases"),
    "grounding_success": ("Grounding success", "submitted plans"),
    "execution_success": ("Execution success", "submitted plans"),
    "postcondition_success": (
        "Postcondition success",
        "cases with symbolic postconditions",
    ),
    "physical_goal_success": (
        "Physical goal success",
        "cases with physical observations",
    ),
    "goal_success": ("Goal success", "cases with required goals"),
    "task_success": ("Task success", "all cases"),
}

MAIN_METRICS = tuple(
    (name, *METRIC_PRESENTATION[name]) for name in SCORING_MAIN_METRICS
)


FAILURE_STAGE_LABELS = {
    "planner_json": "Planner JSON",
    "planner_schema": "Planner schema",
    "planner_guard": "Planner guard",
    "planner_mode": "Planner response mode",
    "planning": "Planner error",
    "world_setup": "World setup",
    "input": "Plan input (IPC)",
    "grounding": "Grounding",
    "execution": "Execution",
    "timeout": "Execution timeout",
    "context_refresh": "Context refresh",
    "executor": "Executor process",
    "goal_verification": "Goal verification",
}

MAIN_METRICS_COLUMNS = ("metric", "successes", "cases", "percent", "denominator")
CLARIFICATION_COLUMNS = ("metric", "value")
FAILURE_COLUMNS = ("failed_stage", "count", "share_of_failures_percent")
FAILED_CASE_COLUMNS = (
    "case_id",
    "failed_stage",
    "planner_outcome",
    "execution_status",
    "error",
)

HEATMAP_FIELDS = (
    "json_valid",
    "schema_valid",
    "guard_valid",
    "planner_success",
    "grounding_success",
    "execution_success",
    "goal_success",
    "task_success",
)

HEATMAP_LABELS = {
    "json_valid": "JSON",
    "schema_valid": "Schema",
    "guard_valid": "Guard",
    "planner_success": "Planner",
    "grounding_success": "Grounding",
    "execution_success": "Execution",
    "goal_success": "Goal",
    "task_success": "Task",
}


def _as_bool(value):
    text = str(value if value is not None else "").strip().casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_results(rows):
    if not rows:
        raise ValueError("results CSV contains no cases")
    required = {"id", "task_success", "failure_stage", "clarification_outcome"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            "results CSV does not use the current evaluation schema; "
            f"missing columns: {', '.join(sorted(missing))}"
        )
    ids = [str(row.get("id", "")).strip() for row in rows]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("results CSV requires non-empty unique case IDs")


def build_main_metrics(rows):
    table = []
    for name, label, denominator in MAIN_METRICS:
        values = [_as_bool(row.get(name)) for row in rows]
        values = [value for value in values if value is not None]
        successes = sum(values)
        percent = round(100.0 * successes / len(values), 1) if values else ""
        table.append(
            {
                "metric": label,
                "successes": successes,
                "cases": len(values),
                "percent": percent,
                "denominator": denominator,
            }
        )
    return table


def build_clarification_table(rows):
    outcomes = Counter(
        str(row.get("clarification_outcome", "")).strip() for row in rows
    )
    tp = outcomes.get("TP", 0)
    fp = outcomes.get("FP", 0)
    fn = outcomes.get("FN", 0)
    tn = outcomes.get("TN", 0)
    precision, recall, f1 = confusion_rates(tp, fp, fn, tn)

    def _percent(value):
        return "n/a" if value is None else f"{100.0 * value:.1f}%"

    scripted = [row for row in rows if str(row.get("clarification_answer", "")).strip()]
    scripted_ok = sum(
        bool(_as_bool(row.get("dialog_resolution_success"))) for row in scripted
    )
    return [
        {"metric": "True positives", "value": tp},
        {"metric": "False positives", "value": fp},
        {"metric": "False negatives", "value": fn},
        {"metric": "True negatives", "value": tn},
        {"metric": "Precision", "value": _percent(precision)},
        {"metric": "Recall", "value": _percent(recall)},
        {"metric": "F1 score", "value": _percent(f1)},
        {
            "metric": "Scripted follow-up success",
            "value": f"{scripted_ok}/{len(scripted)}" if scripted else "n/a",
        },
    ]


def build_failure_distribution(rows):
    failed = [row for row in rows if _as_bool(row.get("task_success")) is False]
    stages = []
    for row in failed:
        stage = str(row.get("failure_stage", "")).strip()
        if stage not in FAILURE_STAGE_LABELS:
            raise ValueError(
                f"case {row.get('id')!r} failed without a known failure_stage"
            )
        stages.append(stage)

    counts = Counter(stages)
    table = []
    for stage in FAILURE_STAGE_LABELS:
        count = counts.get(stage, 0)
        if not count:
            continue
        table.append(
            {
                "failed_stage": FAILURE_STAGE_LABELS[stage],
                "count": count,
                "share_of_failures_percent": round(100.0 * count / len(failed), 1),
            }
        )
    return table


def build_failed_cases(rows):
    table = []
    for row in rows:
        if _as_bool(row.get("task_success")) is not False:
            continue
        stage = str(row.get("failure_stage", "")).strip()
        error = " ".join(str(row.get("error", "")).split())
        if len(error) > 140:
            error = error[:137] + "..."
        table.append(
            {
                "case_id": row.get("id", ""),
                "failed_stage": FAILURE_STAGE_LABELS.get(stage, stage),
                "planner_outcome": row.get("planner_outcome", ""),
                "execution_status": row.get("execution_status", ""),
                "error": error,
            }
        )
    return table


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_figure(figure, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = output_dir / f"{name}.{extension}"
        options = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            options["dpi"] = 200
        figure.savefig(path, **options)
        paths.append(path)
    return paths


def _show_empty_plot(axis, title, message):
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()


def plot_main_metrics(rows, output_dir):
    metrics = [row for row in build_main_metrics(rows) if row["cases"]]
    figure_height = max(4.0, 0.45 * len(metrics) + 1.8)
    figure, axis = plt.subplots(figsize=(9.0, figure_height))

    if not metrics:
        _show_empty_plot(axis, "Main evaluation metrics", "No metrics available")
    else:
        labels = [row["metric"] for row in metrics]
        rates = [row["percent"] for row in metrics]
        bars = axis.barh(labels, rates, color="royalblue")
        axis.invert_yaxis()
        axis.set_xlim(0, 112)
        axis.set_xlabel("Success rate (%)")
        axis.set_title("Main evaluation metrics")
        axis.grid(axis="x", color="lightgray", linewidth=0.7)
        axis.set_axisbelow(True)
        for bar, row in zip(bars, metrics):
            label = f"{row['percent']:.1f}% ({row['successes']}/{row['cases']})"
            axis.text(
                row["percent"] + 1.0,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
            )

    figure.tight_layout()
    paths = _save_figure(figure, Path(output_dir), "main_metrics")
    plt.close(figure)
    return paths


def plot_failure_distribution(rows, output_dir):
    failures = build_failure_distribution(rows)
    figure_height = max(3.5, 0.5 * len(failures) + 1.8)
    figure, axis = plt.subplots(figsize=(9.0, figure_height))

    if not failures:
        _show_empty_plot(axis, "Failure distribution", "No failures recorded")
    else:
        labels = [failure["failed_stage"] for failure in failures]
        counts = [failure["count"] for failure in failures]
        bars = axis.barh(labels, counts, color="firebrick")
        axis.invert_yaxis()
        axis.set_xlabel("Failed cases")
        axis.set_title("Failure distribution")
        axis.xaxis.get_major_locator().set_params(integer=True)
        axis.grid(axis="x", color="lightgray", linewidth=0.7)
        axis.set_axisbelow(True)
        largest_count = max(counts)
        axis.set_xlim(0, max(1.0, largest_count * 1.35))
        for bar, failure in zip(bars, failures):
            label = f"{failure['count']} ({failure['share_of_failures_percent']:.1f}%)"
            axis.text(
                failure["count"] + largest_count * 0.03,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
            )

    figure.tight_layout()
    paths = _save_figure(figure, Path(output_dir), "failure_distribution")
    plt.close(figure)
    return paths


def build_case_stage_matrix(rows):
    case_labels = []
    matrix = []
    for row in rows:
        case_labels.append(str(row.get("id", "")).strip() or "n/a")
        stage_values = []
        for metric in HEATMAP_FIELDS:
            value = _as_bool(row.get(metric))
            if value is True:
                stage_values.append(1)
            elif value is False:
                stage_values.append(0)
            else:
                stage_values.append(-1)
        matrix.append(stage_values)
    return case_labels, matrix


def plot_case_stage_heatmap(rows, output_dir):
    case_labels, matrix = build_case_stage_matrix(rows)
    figure_height = max(4.5, 0.28 * len(case_labels) + 2.2)
    figure, axis = plt.subplots(figsize=(11.0, figure_height))

    if not matrix:
        _show_empty_plot(axis, "Case-stage heatmap", "No cases available")
    else:
        colors = ["lightgray", "firebrick", "seagreen"]
        color_map = ListedColormap(colors)
        color_ranges = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], color_map.N)
        axis.imshow(matrix, aspect="auto", cmap=color_map, norm=color_ranges)
        axis.set_xticks(
            range(len(HEATMAP_FIELDS)),
            [HEATMAP_LABELS[metric] for metric in HEATMAP_FIELDS],
        )
        axis.set_yticks(range(len(case_labels)), case_labels)
        axis.set_xticks(
            [position - 0.5 for position in range(1, len(HEATMAP_FIELDS))],
            minor=True,
        )
        axis.set_yticks(
            [position - 0.5 for position in range(1, len(case_labels))],
            minor=True,
        )
        axis.grid(which="minor", color="white", linewidth=0.5)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_xlabel("Pipeline stage")
        axis.set_ylabel("Case ID")
        axis.set_title("Case-stage heatmap")
        legend = [
            Patch(color="seagreen", label="Passed"),
            Patch(color="firebrick", label="Failed"),
            Patch(color="lightgray", label="Not evaluated"),
        ]
        axis.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    figure.tight_layout()
    paths = _save_figure(figure, Path(output_dir), "case_stage_heatmap")
    plt.close(figure)
    return paths


def generate_plots(rows, output_dir):
    paths = []
    for plot_function in (
        plot_main_metrics,
        plot_failure_distribution,
        plot_case_stage_heatmap,
    ):
        paths.extend(plot_function(rows, output_dir))
    return paths


def generate_report(results_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(results_path)
    check_results(rows)

    metrics = build_main_metrics(rows)
    clarification = build_clarification_table(rows)
    failures = build_failure_distribution(rows)
    failed_cases = build_failed_cases(rows)

    paths = [
        write_csv(output_dir / "main_metrics.csv", metrics, MAIN_METRICS_COLUMNS),
        write_csv(
            output_dir / "clarification.csv", clarification, CLARIFICATION_COLUMNS
        ),
        write_csv(output_dir / "failure_distribution.csv", failures, FAILURE_COLUMNS),
        write_csv(output_dir / "failed_cases.csv", failed_cases, FAILED_CASE_COLUMNS),
    ]
    paths.extend(generate_plots(rows, output_dir / "plots"))
    return tuple(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Create the thesis report tables from evaluation results."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = generate_report(args.results, args.output_dir)
    print("Wrote " + ", ".join(map(str, paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
