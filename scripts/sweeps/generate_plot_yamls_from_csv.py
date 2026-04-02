#!/usr/bin/env python3
"""Generate Stage 1 and Stage 2 plot YAMLs from one or more sweep CSV logs.

Examples:

  ./.venv/bin/python scripts/sweeps/generate_plot_yamls_from_csv.py \
    scripts/sweeps/week3/sweep_jepa_pretrain_finetune_hl1.csv

  ./.venv/bin/python scripts/sweeps/generate_plot_yamls_from_csv.py \
    scripts/sweeps/week3/sweep_jepa_pretrain_finetune_hl1.csv \
    scripts/sweeps/week3/sweep_jepa_pretrain_finetune_hl2.csv

  ./.venv/bin/python scripts/sweeps/generate_plot_yamls_from_csv.py \
    --stage2-suffix=-stage2-fn-fixed \
    scripts/sweeps/week2/sweep_jepa_random_student_rate_stage1_select_log_cropping.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_OUTPUT_PREFIX = "plot_jepa_training_"
DEFAULT_STAGE1_SUFFIX = "-stage1"
DEFAULT_STAGE2_SUFFIX = "-stage2"
DEFAULT_DESCRIPTION_LIMIT = 72


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one or more sweep CSV files and generate two plot YAML manifests per CSV: "
            "one for Stage 1 runs and one for Stage 2 runs."
        )
    )
    parser.add_argument(
        "csv",
        nargs="+",
        help="CSV log files to convert.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix prepended to the CSV stem when naming the generated YAMLs.",
    )
    parser.add_argument(
        "--stage1-suffix",
        default=DEFAULT_STAGE1_SUFFIX,
        help="Suffix appended to each base run_id in the Stage 1 YAML.",
    )
    parser.add_argument(
        "--stage2-suffix",
        default=DEFAULT_STAGE2_SUFFIX,
        help="Suffix appended to each base run_id in the Stage 2 YAML.",
    )
    parser.add_argument(
        "--description-limit",
        type=int,
        default=DEFAULT_DESCRIPTION_LIMIT,
        help="Soft character limit for generated descriptions.",
    )
    return parser.parse_args()


def _resolve_csv_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path.resolve()

    repo_relative = repo_root / raw_path
    if repo_relative.is_file():
        return repo_relative.resolve()

    raise FileNotFoundError("CSV file not found: %s" % raw_path)


def _normalise_run_id(raw_run_id: str) -> str:
    run_id = raw_run_id.strip()
    if not run_id:
        raise ValueError("Encountered an empty run_id value.")
    return run_id


def _choose_value(row: Dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def _short_config_hint(config_path: str) -> str:
    if not config_path:
        return ""

    stem = Path(config_path).stem
    prefixes = (
        "config_jepa_frozen_2drope_qkrms_student_",
        "config_jepa_ema_2drope_qkrms_student_",
        "config_jepa_frozen_2drope_qkrms_",
        "config_jepa_ema_2drope_",
        "config_jepa_frozen_",
        "config_",
    )
    for prefix in prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem


def _build_description(stage_label: str, row: Dict[str, str], limit: int) -> str:
    tokens = [stage_label]

    candidates = [
        ("lr", _choose_value(row, ("lr_value", "lr_max"))),
        ("mr", _choose_value(row, ("mask_value", "mask_rate"))),
        ("mr_s", _choose_value(row, ("student_mask_rate",))),
        ("mr_t", _choose_value(row, ("teacher_mask_rate",))),
        ("hl", _choose_value(row, ("hl_mask_value",))),
        ("str", _choose_value(row, ("strategy",))),
    ]

    for label, value in candidates:
        if value:
            tokens.append("%s=%s" % (label, value))

    config_hint = _short_config_hint((row.get("stage1_config") or "").strip())
    if config_hint:
        tokens.append("cfg=%s" % config_hint)

    description = tokens[0]
    for token in tokens[1:]:
        candidate = "%s %s" % (description, token)
        if len(candidate) > limit and len(description) > len(stage_label):
            break
        description = candidate

    if description == stage_label:
        sweep_name = (row.get("sweep_name") or "").strip()
        if sweep_name:
            description = "%s %s" % (stage_label, sweep_name)

    return description


def _yaml_lines(csv_name: str, entries: List[Tuple[str, str]]) -> List[str]:
    lines = [
        "# Plot config for %s (auto-generated)" % csv_name,
        "train:",
        "  plot:",
    ]
    for run_id, description in entries:
        lines.extend(
            [
                "    %s:" % run_id,
                "      slurm_id: 0",
                "      description: %s" % json.dumps(description),
            ]
        )
    return lines


def _write_yaml(path: Path, csv_name: str, entries: List[Tuple[str, str]]) -> None:
    path.write_text("\n".join(_yaml_lines(csv_name, entries)) + "\n", encoding="utf-8")


def _stage_output_paths(csv_path: Path, output_prefix: str) -> Tuple[Path, Path]:
    stem = "%s%s" % (output_prefix, csv_path.stem)
    return (
        csv_path.with_name("%s_stage_1.yml" % stem),
        csv_path.with_name("%s_stage_2.yml" % stem),
    )


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "run_id" not in (reader.fieldnames or []):
            raise ValueError("CSV file %s does not contain a run_id column." % csv_path)
        return list(reader)


def _build_entries(
    rows: List[Dict[str, str]],
    suffix: str,
    stage_label: str,
    description_limit: int,
) -> List[Tuple[str, str]]:
    entries = []
    for row_index, row in enumerate(rows, start=2):
        try:
            base_run_id = _normalise_run_id(row.get("run_id") or "")
        except ValueError as exc:
            raise ValueError("%s:%d %s" % (row.get("__csv_path__", ""), row_index, exc)) from exc

        entries.append(
            (
                "%s%s" % (base_run_id, suffix),
                _build_description(stage_label, row, description_limit),
            )
        )
    return entries


def _generate_for_csv(
    csv_path: Path,
    output_prefix: str,
    stage1_suffix: str,
    stage2_suffix: str,
    description_limit: int,
) -> Tuple[Path, Path]:
    rows = _load_rows(csv_path)
    for row in rows:
        row["__csv_path__"] = str(csv_path)

    stage1_entries = _build_entries(rows, stage1_suffix, "s1", description_limit)
    stage2_entries = _build_entries(rows, stage2_suffix, "s2", description_limit)

    stage1_path, stage2_path = _stage_output_paths(csv_path, output_prefix)
    _write_yaml(stage1_path, csv_path.name, stage1_entries)
    _write_yaml(stage2_path, csv_path.name, stage2_entries)
    return stage1_path, stage2_path


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    for raw_csv_path in args.csv:
        csv_path = _resolve_csv_path(repo_root, raw_csv_path)
        stage1_path, stage2_path = _generate_for_csv(
            csv_path=csv_path,
            output_prefix=args.output_prefix,
            stage1_suffix=args.stage1_suffix,
            stage2_suffix=args.stage2_suffix,
            description_limit=args.description_limit,
        )
        print("Generated %s" % stage1_path)
        print("Generated %s" % stage2_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
