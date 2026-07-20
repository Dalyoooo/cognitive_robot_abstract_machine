import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks")]

from .report import (  # noqa: E402
    build_main_metrics,
    check_results,
    read_csv,
    write_csv,
    write_latex,
)

COMPARISON_COLUMNS = ("metric", "base", "finetuned", "difference_pp", "denominator")


def _format_rate(row):
    """Format one Table A row as "successes/cases (percent%)"."""
    if not row["cases"]:
        return "n/a"
    return f"{row['successes']}/{row['cases']} ({row['percent']:.1f}%)"


def build_comparison(base_rows, finetuned_rows):
    """Combine two Table A results into one comparison table."""
    base_metrics = build_main_metrics(base_rows)
    finetuned_metrics = build_main_metrics(finetuned_rows)

    table = []
    for base, finetuned in zip(base_metrics, finetuned_metrics):
        if base["cases"] and finetuned["cases"]:
            difference = round(finetuned["percent"] - base["percent"], 1)
        else:
            difference = ""
        table.append(
            {
                "metric": base["metric"],
                "base": _format_rate(base),
                "finetuned": _format_rate(finetuned),
                "difference_pp": difference,
                "denominator": base["denominator"],
            }
        )
    return table


def main():
    """Write the base vs fine-tuned comparison table."""
    parser = argparse.ArgumentParser(
        description="Compare a base and a fine-tuned evaluation run."
    )
    parser.add_argument("--base", type=Path, required=True, help="Base results.csv")
    parser.add_argument(
        "--finetuned", type=Path, required=True, help="Fine-tuned results.csv"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_csv(args.base)
    finetuned_rows = read_csv(args.finetuned)
    check_results(base_rows)
    check_results(finetuned_rows)

    table = build_comparison(base_rows, finetuned_rows)
    paths = (
        write_csv(args.output_dir / "comparison.csv", table, COMPARISON_COLUMNS),
        write_latex(
            args.output_dir / "tables" / "comparison.tex", table, COMPARISON_COLUMNS
        ),
    )
    print("Wrote " + ", ".join(map(str, paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
