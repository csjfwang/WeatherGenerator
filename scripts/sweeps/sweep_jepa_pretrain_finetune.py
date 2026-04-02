#!/usr/bin/env python3
"""Unified two-stage JEPA sweep for pretraining and finetuning.

This script samples Stage 1 overlay configs, then submits the original two-stage
training chain through `launch-slurm-multi.py`:

  Stage 1: pretraining from `--stage1-config`
  Stage 2: finetuning from `--finetune-config`

By default it sweeps:
  - learning rate at `training_config.learning_rate_scheduling.lr_max`
  - masking rate at `training_config.model_input.random_easy.masking_strategy_config.rate`

and keeps the teacher/full-field rate fixed at:
  - `training_config.target_input.random_easy_target.masking_strategy_config.rate = 1.0`

Optionally it can also sweep:
  - `training_config.model_input.random_easy.masking_strategy_config.hl_mask`

Examples:

  Dry run 3 experiments with the default full-ERA5 configs:
    ./scripts/sweeps/sweep_jepa_pretrain_finetune.py 3 --dry-run

  Submit 10 experiments with a healpix Stage 1 config:
    ./scripts/sweeps/sweep_jepa_pretrain_finetune.py 10 \
      --stage1-config config/config_jepa_frozen_2drope_qkrms_student_healpix.yml

  Submit q-only Stage 1 + q-only finetuning:
    ./scripts/sweeps/sweep_jepa_pretrain_finetune.py 8 \
      --stage1-config config/config_jepa_ema_2drope_qkrms_random.yml \
      --finetune-config config/config_jepa_finetuning_q_abl_stage2.yml

  Sweep a different mask path:
    ./scripts/sweeps/sweep_jepa_pretrain_finetune.py 5 \
      --mask-path training_config.target_input.random_easy_target.masking_strategy_config.rate \
      --no-teacher-mask-override

  Sweep healpix `hl_mask` between 1 and 4:
    ./scripts/sweeps/sweep_jepa_pretrain_finetune.py 8 \
      --stage1-config config/week3/config_jepa_frozen_2drope_qkrms_student_healpix_1.yml \
      --hl-mask-min 1 \
      --hl-mask-max 4

  Add extra fixed overrides to the generated Stage 1 overlay:
    ./scripts/sweeps/sweep_jepa_pretrain_finetune.py 4 \
      --set training_config.samples_per_mini_epoch=1024 \
      --set wgtags.variant=random_baseline

Notes:
  - `--dry-run` prints commands without submitting them.
  - Unknown extra CLI args are forwarded to `launch-slurm-multi.py`.
  - A CSV manifest and generated overlay YAML files are written for every run.
"""

import argparse
import csv
import math
import os
import random
import re
import shlex
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_SWEEP_NAME = "jepa_pretrain_finetune"
DEFAULT_STAGE1_CONFIG = "config/week3/config_jepa_frozen_2drope_qkrms_student_healpix.yml"
DEFAULT_FINETUNE_CONFIG = "config/config_jepa_finetuning.yml"
DEFAULT_LR_PATH = "training_config.learning_rate_scheduling.lr_max"
DEFAULT_MASK_PATH = "training_config.model_input.random_easy.masking_strategy_config.rate"
DEFAULT_HL_MASK_PATH = "training_config.model_input.random_easy.masking_strategy_config.hl_mask"
DEFAULT_TEACHER_MASK_PATH = (
    "training_config.target_input.random_easy_target.masking_strategy_config.rate"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_launcher(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "launch-slurm-multi.py"


def _default_logs_dir(repo_root: Path) -> Path:
    return repo_root / "scripts" / "sweeps" / "logs"


def _default_config_runs_dir(repo_root: Path) -> Path:
    return repo_root / "config" / "sweep_runs"


def _shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _git_deleted_tracked_files(repo_root: Path) -> List[str]:
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return []

    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--deleted"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Sample Stage 1 pretraining overlays and submit chained Stage 1 + Stage 2 "
            "JEPA jobs through launch-slurm-multi.py."
        ),
    )
    parser.add_argument(
        "num_experiments",
        nargs="?",
        type=int,
        default=10,
        help="Number of experiments to generate and submit. Default: 10",
    )
    parser.add_argument(
        "--stage1-config",
        default=DEFAULT_STAGE1_CONFIG,
        help="Base config used for Stage 1 pretraining.",
    )
    parser.add_argument(
        "--finetune-config",
        default=DEFAULT_FINETUNE_CONFIG,
        help="Config used for Stage 2 finetuning.",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=_default_launcher(repo_root),
        help="Path to WeatherGenerator-private/hpc/launch-slurm-multi.py.",
    )
    parser.add_argument(
        "--sweep-name",
        default=DEFAULT_SWEEP_NAME,
        help="Value written to wgtags.exp and to the CSV manifest.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to pass to launch-slurm-multi.py.",
    )
    parser.add_argument(
        "--stage1-jobs",
        type=int,
        default=1,
        help="Number of chained Stage 1 jobs. Default: 1",
    )
    parser.add_argument(
        "--stage2-jobs",
        type=int,
        default=1,
        help="Number of chained Stage 2 jobs. Default: 1",
    )
    parser.add_argument(
        "--run-id-length",
        type=int,
        default=8,
        help="Length of the random run_id suffix. Default: 8",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=_default_logs_dir(repo_root),
        help="Directory for the CSV manifest and debug log.",
    )
    parser.add_argument(
        "--config-output-dir",
        type=Path,
        default=_default_config_runs_dir(repo_root),
        help="Directory where generated overlay YAML files are written.",
    )
    parser.add_argument(
        "--log-stem",
        default="sweep_jepa_pretrain_finetune",
        help="Base filename stem for the CSV manifest and debug log.",
    )
    parser.add_argument(
        "--lr-path",
        default=DEFAULT_LR_PATH,
        help="Dotted config path for the learning-rate sweep.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum learning-rate value.",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=5e-5,
        help="Maximum learning-rate value.",
    )
    parser.add_argument(
        "--lr-distribution",
        choices=("log", "uniform"),
        default="log",
        help="Distribution used to sample the learning rate.",
    )
    parser.add_argument(
        "--mask-path",
        default=DEFAULT_MASK_PATH,
        help="Dotted config path for the masking-rate sweep.",
    )
    parser.add_argument(
        "--mask-min",
        type=float,
        default=0.10,
        help="Minimum masking-rate value.",
    )
    parser.add_argument(
        "--mask-max",
        type=float,
        default=0.90,
        help="Maximum masking-rate value.",
    )
    parser.add_argument(
        "--mask-distribution",
        choices=("uniform", "log"),
        default="uniform",
        help="Distribution used to sample the masking rate.",
    )
    parser.add_argument(
        "--mask-precision",
        type=int,
        default=2,
        help="Decimal places kept for the masking-rate sample. Default: 2",
    )
    parser.add_argument(
        "--teacher-mask-path",
        default=DEFAULT_TEACHER_MASK_PATH,
        help="Dotted config path for the fixed teacher/full-field mask rate.",
    )
    parser.add_argument(
        "--teacher-mask-value",
        type=float,
        default=1.0,
        help="Fixed teacher/full-field mask rate. Default: 1.0",
    )
    parser.add_argument(
        "--hl-mask-path",
        default=DEFAULT_HL_MASK_PATH,
        help="Dotted config path for an optional integer hl_mask sweep.",
    )
    parser.add_argument(
        "--hl-mask-min",
        type=int,
        default=None,
        help="Minimum hl_mask value. Enables hl_mask sweeping when set with --hl-mask-max.",
    )
    parser.add_argument(
        "--hl-mask-max",
        type=int,
        default=None,
        help="Maximum hl_mask value. Enables hl_mask sweeping when set with --hl-mask-min.",
    )
    parser.add_argument(
        "--no-teacher-mask-override",
        action="store_true",
        help="Do not inject the fixed teacher/full-field mask path into the overlay.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Extra fixed overlay values to add. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra wgtags entries to add. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and generated config paths without submitting jobs.",
    )

    args, launcher_args = parser.parse_known_args()
    return args, launcher_args


def _resolve_input_path(repo_root: Path, value: str) -> Path:
    candidates = [Path(value), repo_root / value]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("File not found: %s" % value)


def _launcher_config_arg(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    repo_root_resolved = repo_root.resolve()
    try:
        relative = resolved.relative_to(repo_root_resolved)
        return "./%s" % relative.as_posix()
    except ValueError:
        return str(resolved)


def _resolve_unique_file(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    idx = 1
    while True:
        candidate = parent / ("%s_%02d%s" % (stem, idx, suffix))
        if not candidate.exists():
            return candidate
        idx += 1


def _log(message: str, debug_log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (timestamp, message)
    print(line)
    with debug_log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _sample_value(min_value: float, max_value: float, distribution: str) -> float:
    if min_value <= 0.0 and distribution == "log":
        raise ValueError("Log-uniform sampling requires positive bounds.")

    if distribution == "log":
        return math.exp(random.uniform(math.log(min_value), math.log(max_value)))
    return random.uniform(min_value, max_value)


def _generate_run_id(length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _sample_int_value(min_value: int, max_value: int) -> int:
    return random.randint(min_value, max_value)


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None

    if re.match(r"^[+-]?[0-9]+$", value):
        try:
            return int(value)
        except ValueError:
            pass

    try:
        return float(value)
    except ValueError:
        return value


def _parse_assignment(raw_assignment: str) -> Tuple[str, Any]:
    if "=" not in raw_assignment:
        raise ValueError("Expected PATH=VALUE assignment, got '%s'" % raw_assignment)
    key, value = raw_assignment.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Assignment is missing a config path: '%s'" % raw_assignment)
    return key, _parse_scalar(value)


def _set_nested(mapping: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("Empty config path.")

    current = mapping
    for part in parts[:-1]:
        next_value = current.get(part)
        if next_value is None:
            next_value = {}
            current[part] = next_value
        elif not isinstance(next_value, dict):
            raise ValueError("Cannot nest under non-mapping path '%s'" % dotted_path)
        current = next_value

    current[parts[-1]] = value


def _quote_yaml_string(value: str) -> str:
    return "'%s'" % value.replace("'", "''")


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return _quote_yaml_string(str(value))


def _dump_yaml_lines(value: Any, indent: int) -> List[str]:
    prefix = " " * indent
    lines = []  # type: List[str]

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append("%s%s:" % (prefix, key))
                lines.extend(_dump_yaml_lines(item, indent + 2))
            else:
                lines.append("%s%s: %s" % (prefix, key, _format_yaml_scalar(item)))
        return lines

    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append("%s-" % prefix)
                lines.extend(_dump_yaml_lines(item, indent + 2))
            else:
                lines.append("%s- %s" % (prefix, _format_yaml_scalar(item)))
        return lines

    lines.append("%s%s" % (prefix, _format_yaml_scalar(value)))
    return lines


def _write_overlay_config(path: Path, overlay: Dict[str, Any]) -> None:
    lines = [
        "# Auto-generated Stage 1 overlay for the unified pretrain+finetune sweep.",
        "",
    ]
    lines.extend(_dump_yaml_lines(overlay, 0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_overlay(
    args: argparse.Namespace,
    lr_value: float,
    mask_value: float,
    hl_mask_value: Optional[int],
) -> Dict[str, Any]:
    overlay = {}  # type: Dict[str, Any]

    _set_nested(overlay, args.lr_path, lr_value)
    _set_nested(overlay, args.mask_path, mask_value)
    if hl_mask_value is not None:
        _set_nested(overlay, args.hl_mask_path, hl_mask_value)

    if not args.no_teacher_mask_override:
        _set_nested(overlay, args.teacher_mask_path, args.teacher_mask_value)

    for raw_assignment in args.set:
        path, value = _parse_assignment(raw_assignment)
        _set_nested(overlay, path, value)

    wgtags = overlay.get("wgtags")
    if wgtags is None:
        wgtags = {}
        overlay["wgtags"] = wgtags
    elif not isinstance(wgtags, dict):
        raise ValueError("wgtags must be a mapping if overridden via --set.")

    wgtags["exp"] = args.sweep_name
    wgtags["sweep_script"] = "sweep_jepa_pretrain_finetune.py"
    for raw_tag in args.tag:
        key, value = _parse_assignment(raw_tag)
        wgtags[key] = value

    return overlay


def _csv_fieldnames() -> List[str]:
    return [
        "run_id",
        "lr_value",
        "mask_value",
        "hl_mask_value",
        "sweep_name",
        "stage1_config",
        "finetune_config",
        "overlay_config",
        "lr_path",
        "mask_path",
        "hl_mask_path",
    ]


def main() -> int:
    args, launcher_args = _parse_args()
    repo_root = _repo_root()

    if args.num_experiments <= 0:
        raise SystemExit("num_experiments must be positive")
    if args.run_id_length <= 0:
        raise SystemExit("--run-id-length must be positive")
    if (args.hl_mask_min is None) != (args.hl_mask_max is None):
        raise SystemExit("--hl-mask-min and --hl-mask-max must be provided together")
    if args.hl_mask_min is not None and args.hl_mask_min > args.hl_mask_max:
        raise SystemExit("--hl-mask-min must be <= --hl-mask-max")

    deleted_tracked_files = _git_deleted_tracked_files(repo_root)
    if deleted_tracked_files:
        preview = deleted_tracked_files[:10]
        preview_text = "\n".join("  - %s" % path for path in preview)
        remaining = len(deleted_tracked_files) - len(preview)
        if remaining > 0:
            preview_text += "\n  - ... (%d more)" % remaining

        raise SystemExit(
            "Cannot submit the unified sweep because the repo has deleted tracked files.\n"
            "launch-slurm-multi.py copies the repo from `git ls-files`, and that copy step "
            "fails when tracked files are missing from the working tree.\n"
            "Deleted tracked files detected:\n"
            "%s\n"
            "To keep the deletions, stage them first, for example:\n"
            "  git add -u config\n"
            "Or restore the deleted files before retrying."
            % preview_text
        )

    stage1_config = _resolve_input_path(repo_root, args.stage1_config)
    finetune_config = _resolve_input_path(repo_root, args.finetune_config)
    launcher = args.launcher.resolve()

    if not launcher.is_file():
        raise FileNotFoundError("Launcher not found: %s" % launcher)

    logs_dir = args.logs_dir.resolve()
    config_output_dir = args.config_output_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = _resolve_unique_file(logs_dir / ("%s_%s.csv" % (args.log_stem, timestamp)))
    debug_log_path = _resolve_unique_file(logs_dir / ("%s_%s.log" % (args.log_stem, timestamp)))

    with csv_log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
        writer.writeheader()

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", os.path.join(env["HOME"], ".cache", "uv"))

    stage1_arg = _launcher_config_arg(repo_root, stage1_config)
    finetune_arg = _launcher_config_arg(repo_root, finetune_config)

    _log("=== Unified sweep started ===", debug_log_path)
    _log("Working directory: %s" % os.getcwd(), debug_log_path)
    _log("Launcher: %s" % launcher, debug_log_path)
    _log("Stage 1 config: %s" % stage1_config, debug_log_path)
    _log("Finetune config: %s" % finetune_config, debug_log_path)
    _log("CSV manifest: %s" % csv_log_path, debug_log_path)
    _log("Debug log: %s" % debug_log_path, debug_log_path)
    _log("Config output dir: %s" % config_output_dir, debug_log_path)
    _log(
        "Sweeping lr via %s in [%s, %s] (%s)"
        % (args.lr_path, args.lr_min, args.lr_max, args.lr_distribution),
        debug_log_path,
    )
    _log(
        "Sweeping mask via %s in [%s, %s] (%s)"
        % (args.mask_path, args.mask_min, args.mask_max, args.mask_distribution),
        debug_log_path,
    )
    if args.hl_mask_min is not None:
        _log(
            "Sweeping hl_mask via %s in [%d, %d] (integer uniform)"
            % (args.hl_mask_path, args.hl_mask_min, args.hl_mask_max),
            debug_log_path,
        )
    if args.no_teacher_mask_override:
        _log("Teacher mask override disabled.", debug_log_path)
    else:
        _log(
            "Teacher mask fixed via %s = %s"
            % (args.teacher_mask_path, args.teacher_mask_value),
            debug_log_path,
        )
    if args.set:
        _log("Extra fixed overrides: %s" % ", ".join(args.set), debug_log_path)
    if args.tag:
        _log("Extra wgtags: %s" % ", ".join(args.tag), debug_log_path)
    if launcher_args:
        _log("Forwarded launcher args: %s" % _shell_join(launcher_args), debug_log_path)
    if args.dry_run:
        _log("Dry-run mode enabled. No jobs will be submitted.", debug_log_path)

    failures = []  # type: List[Tuple[str, int]]

    for index in range(1, args.num_experiments + 1):
        run_id = _generate_run_id(args.run_id_length)
        lr_value = _sample_value(args.lr_min, args.lr_max, args.lr_distribution)
        mask_value = round(
            _sample_value(args.mask_min, args.mask_max, args.mask_distribution),
            args.mask_precision,
        )
        hl_mask_value = None  # type: Optional[int]
        if args.hl_mask_min is not None:
            hl_mask_value = _sample_int_value(args.hl_mask_min, args.hl_mask_max)

        overlay = _build_overlay(args, lr_value, mask_value, hl_mask_value)
        overlay_path = config_output_dir / (
            "pretrain_finetune_overlay_%s_%02d_%s.yml" % (timestamp, index, run_id)
        )
        _write_overlay_config(overlay_path, overlay)
        overlay_arg = _launcher_config_arg(repo_root, overlay_path)

        with csv_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
            writer.writerow(
                {
                    "run_id": run_id,
                    "lr_value": "%.2e" % lr_value,
                    "mask_value": ("%." + str(args.mask_precision) + "f") % mask_value,
                    "hl_mask_value": "" if hl_mask_value is None else str(hl_mask_value),
                    "sweep_name": args.sweep_name,
                    "stage1_config": stage1_arg,
                    "finetune_config": finetune_arg,
                    "overlay_config": overlay_arg,
                    "lr_path": args.lr_path,
                    "mask_path": args.mask_path,
                    "hl_mask_path": "" if hl_mask_value is None else args.hl_mask_path,
                }
            )

        command = [
            str(launcher),
            "--run-id",
            run_id,
            "--chain-jobs",
            str(args.stage1_jobs),
            str(args.stage2_jobs),
            "--base-config",
            stage1_arg,
            "--config",
            overlay_arg,
            finetune_arg,
            "--nodes",
            str(args.nodes),
        ]
        command.extend(launcher_args)

        _log("--- Experiment %d/%d [%s] ---" % (index, args.num_experiments, run_id), debug_log_path)
        _log("lr_value       = %.2e" % lr_value, debug_log_path)
        _log(
            "mask_value     = %s"
            % (("%." + str(args.mask_precision) + "f") % mask_value),
            debug_log_path,
        )
        if hl_mask_value is not None:
            _log("hl_mask_value  = %d" % hl_mask_value, debug_log_path)
        _log("overlay_config = %s" % overlay_path, debug_log_path)
        _log("launch command = %s" % _shell_join(command), debug_log_path)

        if args.dry_run:
            continue

        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
        )
        if result.stdout:
            with debug_log_path.open("a", encoding="utf-8") as handle:
                handle.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    handle.write("\n")
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

        if result.returncode != 0:
            failures.append((run_id, result.returncode))
            _log(
                "ERROR: submission failed for %s with exit code %d"
                % (run_id, result.returncode),
                debug_log_path,
            )
        else:
            _log(
                "Submitted successfully. Expected outputs: output/%s-stage1 and output/%s-stage2"
                % (run_id, run_id),
                debug_log_path,
            )

    _log("=== Unified sweep complete ===", debug_log_path)
    _log("Experiments prepared: %d" % args.num_experiments, debug_log_path)
    _log("CSV manifest: %s" % csv_log_path, debug_log_path)

    if failures:
        _log("Submission failures: %d" % len(failures), debug_log_path)
        for run_id, returncode in failures:
            _log("  - %s: exit code %d" % (run_id, returncode), debug_log_path)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
