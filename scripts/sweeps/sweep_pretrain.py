#!/usr/bin/env python3
"""Flexible pretraining sweep with optional chained finetuning.

By default submits Stage 1 (pretraining) jobs through `launch-slurm.py`.
When --finetune-config is given, chains to a Stage 2 finetuning job
using `launch-slurm-multi.py` instead.

Sweep parameters are fully user-defined: any config path at any nesting
level can be swept over an explicit list of values. Pass --param once
per parameter, or collect parameters in a JSON spec file (--spec).
Multiple team members can each pass their own --param flags or point to
their own spec file without modifying the script.

Sweep spec JSON format (--spec):
  {
    "params": [
      {"path": "training_config.learning_rate_scheduling.lr_max",
       "values": [1e-5, 2e-5, 5e-5, 1e-4]},
      {"path": "training_config.model_input.mixed.masking_strategy_config.rate",
       "values": [0.3, 0.5, 0.7]}
    ],
    "fixed": [
      {"path": "training_config.samples_per_mini_epoch", "value": 1024}
    ]
  }

Examples:

  Dry run 5 experiments sweeping lr and masking rate:
    ./scripts/sweeps/sweep_pretrain.py 5 --dry-run \\
      --param "training_config.learning_rate_scheduling.lr_max=1e-5,2e-5,5e-5" \\
      --param "training_config.model_input.mixed.masking_strategy_config.rate=0.3,0.5,0.7"

  Use a spec file (values defined outside the script):
    ./scripts/sweeps/sweep_pretrain.py 10 --spec scripts/sweeps/my_sweep.json

  Add fixed overrides on top of the spec:
    ./scripts/sweeps/sweep_pretrain.py 4 \\
      --spec scripts/sweeps/my_sweep.json \\
      --set wgtags.variant=swath_test \\
      --set training_config.samples_per_mini_epoch=2048

  Submit with chained finetuning (uses launch-slurm-multi.py):
    ./scripts/sweeps/sweep_pretrain.py 8 \\
      --base-config config/config_mae.yml \\
      --finetune-config config/config_mae_forecast_finetuning.yml \\
      --param "training_config.learning_rate_scheduling.lr_max=1e-5,5e-5,1e-4"

Notes:
  - --param values are comma-separated scalars; commas inside values are not supported.
  - --param and --spec can be combined; --param flags are appended after spec params.
  - Unknown extra CLI args are forwarded unchanged to the launcher.
  - A CSV manifest and generated overlay YAML files are written for every run.
"""

import argparse
import csv
import json
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


DEFAULT_SWEEP_NAME = "pretrain"
DEFAULT_BASE_CONFIG = "config/config_mae.yml"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _single_launcher(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "launch-slurm.py"


def _multi_launcher(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "launch-slurm-multi.py"


def _default_logs_dir(repo_root: Path) -> Path:
    return repo_root / "scripts" / "sweeps" / "logs"


def _default_config_runs_dir(repo_root: Path) -> Path:
    return repo_root / "config" / "sweep_runs"


# ---------------------------------------------------------------------------
# Sweep parameter definition
# ---------------------------------------------------------------------------


class SweepParam:
    """A config path with a list of candidate values to sample from."""

    def __init__(self, path: str, values: List[Any]) -> None:
        if not path:
            raise ValueError("SweepParam path must be non-empty.")
        if not values:
            raise ValueError("SweepParam %r has no values." % path)
        self.path = path
        self.values = values

    def sample(self) -> Any:
        return random.choice(self.values)


def _parse_param_flag(raw: str) -> SweepParam:
    """Parse a '--param PATH=v1,v2,v3' string into a SweepParam."""
    if "=" not in raw:
        raise ValueError("--param must be PATH=val1,val2,... — got: %r" % raw)
    path, raw_values = raw.split("=", 1)
    path = path.strip()
    if not path:
        raise ValueError("--param is missing the config path in: %r" % raw)
    values = [_parse_scalar(v.strip()) for v in raw_values.split(",") if v.strip()]
    if not values:
        raise ValueError("--param %r has an empty value list." % path)
    return SweepParam(path, values)


def _load_spec(spec_path: Path) -> Tuple[List[SweepParam], List[Tuple[str, Any]]]:
    """Load sweep params and fixed overrides from a JSON spec file."""
    with spec_path.open(encoding="utf-8") as handle:
        spec = json.load(handle)

    params = []
    for entry in spec.get("params", []):
        params.append(SweepParam(entry["path"], entry["values"]))

    fixed = []
    for entry in spec.get("fixed", []):
        fixed.append((entry["path"], entry["value"]))

    return params, fixed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Submit pretraining sweep jobs with fully user-defined sweep parameters.\n"
            "Optionally chains to a finetuning stage via launch-slurm-multi.py."
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
        "--base-config",
        default=DEFAULT_BASE_CONFIG,
        help="Base config for pretraining (Stage 1). Default: %s" % DEFAULT_BASE_CONFIG,
    )
    parser.add_argument(
        "--finetune-config",
        default=None,
        metavar="CONFIG",
        help=(
            "Config for Stage 2 finetuning. When provided, uses launch-slurm-multi.py "
            "to chain both stages. Omit to run Stage 1 only via launch-slurm.py."
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        metavar="JSON_FILE",
        help=(
            "JSON file defining params and fixed overrides for the sweep. "
            "See script docstring for the format."
        ),
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="PATH=v1,v2,...",
        help=(
            "Config path and comma-separated list of values to sweep. "
            "Can be passed multiple times. Appended after any --spec params."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Fixed overlay value added to every experiment. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra wgtags entries added to every experiment. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sweep-name",
        default=DEFAULT_SWEEP_NAME,
        help="Value written to wgtags.exp and to the CSV manifest. Default: %s" % DEFAULT_SWEEP_NAME,
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes passed to the launcher. Default: 1",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="SLURM account passed to the launcher (e.g. haicore-project1).",
    )
    parser.add_argument(
        "--stage1-jobs",
        type=int,
        default=1,
        help="Number of chained Stage 1 jobs (multi-launcher only). Default: 1",
    )
    parser.add_argument(
        "--stage2-jobs",
        type=int,
        default=1,
        help="Number of chained Stage 2 jobs (multi-launcher only). Default: 1",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=None,
        help=(
            "Explicit path to the launcher script. By default uses launch-slurm.py "
            "(single stage) or launch-slurm-multi.py (when --finetune-config is set)."
        ),
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
        help=(
            "Directory where generated overlay YAML files are written. "
            "Must be under config/ so the launcher copies it to the Slurm node."
        ),
    )
    parser.add_argument(
        "--log-stem",
        default="sweep_pretrain",
        help="Base filename stem for the CSV manifest and debug log. Default: sweep_pretrain",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and generated config paths without submitting jobs.",
    )

    args, launcher_args = parser.parse_known_args()
    return args, launcher_args


# ---------------------------------------------------------------------------
# Utilities (logging, git, path resolution, ID generation)
# ---------------------------------------------------------------------------


def _shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _log(message: str, debug_log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (timestamp, message)
    print(line)
    with debug_log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _generate_run_id(length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _resolve_input_path(repo_root: Path, value: str) -> Path:
    for candidate in (Path(value), repo_root / value):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("File not found: %s" % value)


def _launcher_config_arg(repo_root: Path, path: Path) -> str:
    """Return a config path as a relative string if inside the repo, absolute otherwise."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
        return "./%s" % relative.as_posix()
    except ValueError:
        return str(resolved)


def _resolve_unique_file(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem, suffix, parent = base_path.stem, base_path.suffix, base_path.parent
    idx = 1
    while True:
        candidate = parent / ("%s_%02d%s" % (stem, idx, suffix))
        if not candidate.exists():
            return candidate
        idx += 1


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


# ---------------------------------------------------------------------------
# Scalar parsing
# ---------------------------------------------------------------------------


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
        raise ValueError("Expected PATH=VALUE assignment, got %r" % raw_assignment)
    key, value = raw_assignment.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Assignment is missing a config path: %r" % raw_assignment)
    return key, _parse_scalar(value)


# ---------------------------------------------------------------------------
# Config overlay building and YAML writing
# ---------------------------------------------------------------------------


def _set_nested(mapping: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [p for p in dotted_path.split(".") if p]
    if not parts:
        raise ValueError("Empty config path.")
    current = mapping
    for part in parts[:-1]:
        next_value = current.get(part)
        if next_value is None:
            next_value = {}
            current[part] = next_value
        elif not isinstance(next_value, dict):
            raise ValueError("Cannot nest under non-mapping path %r" % dotted_path)
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
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append("%s-" % prefix)
                lines.extend(_dump_yaml_lines(item, indent + 2))
            else:
                lines.append("%s- %s" % (prefix, _format_yaml_scalar(item)))
    else:
        lines.append("%s%s" % (prefix, _format_yaml_scalar(value)))
    return lines


def _write_overlay_config(path: Path, overlay: Dict[str, Any]) -> None:
    lines = ["# Auto-generated overlay for sweep_pretrain.py", ""]
    lines.extend(_dump_yaml_lines(overlay, 0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_overlay(
    sweep_params: List[SweepParam],
    sampled: Dict[str, Any],
    fixed_overrides: List[Tuple[str, Any]],
    extra_sets: List[str],
    extra_tags: List[str],
    sweep_name: str,
) -> Dict[str, Any]:
    overlay = {}  # type: Dict[str, Any]

    for param in sweep_params:
        _set_nested(overlay, param.path, sampled[param.path])

    for path, value in fixed_overrides:
        _set_nested(overlay, path, value)

    for raw in extra_sets:
        path, value = _parse_assignment(raw)
        _set_nested(overlay, path, value)

    wgtags = overlay.setdefault("wgtags", {})
    if not isinstance(wgtags, dict):
        raise ValueError("wgtags must be a mapping if overridden via --set.")
    wgtags["exp"] = sweep_name
    wgtags["sweep_script"] = "sweep_pretrain.py"
    for raw in extra_tags:
        key, value = _parse_assignment(raw)
        wgtags[key] = value

    return overlay


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _csv_fieldnames(sweep_params: List[SweepParam], with_finetune: bool) -> List[str]:
    fields = ["run_id", "sweep_name", "base_config", "overlay_config"]
    fields += [p.path for p in sweep_params]
    if with_finetune:
        fields.append("finetune_config")
    return fields


def _csv_row(
    run_id: str,
    sweep_name: str,
    base_config_arg: str,
    overlay_arg: str,
    sweep_params: List[SweepParam],
    sampled: Dict[str, Any],
    finetune_arg: Optional[str],
) -> Dict[str, Any]:
    row = {
        "run_id": run_id,
        "sweep_name": sweep_name,
        "base_config": base_config_arg,
        "overlay_config": overlay_arg,
    }
    for param in sweep_params:
        value = sampled[param.path]
        row[param.path] = repr(value) if isinstance(value, float) else str(value)
    if finetune_arg is not None:
        row["finetune_config"] = finetune_arg
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args, launcher_args = _parse_args()
    repo_root = _repo_root()

    if args.num_experiments <= 0:
        raise SystemExit("num_experiments must be positive.")
    if args.run_id_length <= 0:
        raise SystemExit("--run-id-length must be positive.")

    # --- resolve sweep params ---
    spec_params = []  # type: List[SweepParam]
    fixed_from_spec = []  # type: List[Tuple[str, Any]]
    if args.spec is not None:
        spec_path = args.spec.resolve()
        if not spec_path.is_file():
            raise FileNotFoundError("Spec file not found: %s" % spec_path)
        spec_params, fixed_from_spec = _load_spec(spec_path)

    cli_params = [_parse_param_flag(raw) for raw in args.param]
    sweep_params = spec_params + cli_params

    if not sweep_params:
        raise SystemExit(
            "No sweep parameters defined. Use --param PATH=v1,v2,... or --spec FILE."
        )

    # --- resolve launcher ---
    with_finetune = args.finetune_config is not None
    if args.launcher is not None:
        launcher = args.launcher.resolve()
    elif with_finetune:
        launcher = _multi_launcher(repo_root).resolve()
    else:
        launcher = _single_launcher(repo_root).resolve()

    if not launcher.is_file():
        raise FileNotFoundError("Launcher not found: %s" % launcher)

    # --- resolve config paths ---
    base_config = _resolve_input_path(repo_root, args.base_config)
    finetune_config = None  # type: Optional[Path]
    if with_finetune:
        finetune_config = _resolve_input_path(repo_root, args.finetune_config)

    # --- check for deleted tracked files (launcher copies via git ls-files) ---
    deleted = _git_deleted_tracked_files(repo_root)
    if deleted:
        preview = "\n".join("  - %s" % f for f in deleted[:10])
        if len(deleted) > 10:
            preview += "\n  - ... (%d more)" % (len(deleted) - 10)
        raise SystemExit(
            "Cannot submit: the repo has deleted tracked files.\n"
            "The launcher copies the repo from `git ls-files` and fails on missing files.\n"
            "%s\nStage the deletions first (e.g. git add -u config) then retry." % preview
        )

    # --- set up output directories and log files ---
    logs_dir = args.logs_dir.resolve()
    config_output_dir = args.config_output_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = _resolve_unique_file(logs_dir / ("%s_%s.csv" % (args.log_stem, timestamp)))
    debug_log_path = _resolve_unique_file(logs_dir / ("%s_%s.log" % (args.log_stem, timestamp)))

    fieldnames = _csv_fieldnames(sweep_params, with_finetune)
    with csv_log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", os.path.join(env["HOME"], ".cache", "uv"))

    base_config_arg = _launcher_config_arg(repo_root, base_config)
    finetune_arg = (
        _launcher_config_arg(repo_root, finetune_config) if finetune_config is not None else None
    )

    # --- log sweep setup ---
    _log("=== Pretrain sweep started ===", debug_log_path)
    _log("Launcher: %s" % launcher, debug_log_path)
    _log("Mode: %s" % ("pretrain + finetune" if with_finetune else "pretrain only"), debug_log_path)
    _log("Base config: %s" % base_config, debug_log_path)
    if finetune_config:
        _log("Finetune config: %s" % finetune_config, debug_log_path)
    _log("CSV manifest: %s" % csv_log_path, debug_log_path)
    _log("Config output dir: %s" % config_output_dir, debug_log_path)
    _log("Sweep parameters:", debug_log_path)
    for param in sweep_params:
        _log("  %s  →  %s" % (param.path, param.values), debug_log_path)
    if args.set:
        _log("Fixed overrides: %s" % ", ".join(args.set), debug_log_path)
    if args.tag:
        _log("Extra wgtags: %s" % ", ".join(args.tag), debug_log_path)
    if launcher_args:
        _log("Forwarded launcher args: %s" % _shell_join(launcher_args), debug_log_path)
    if args.dry_run:
        _log("Dry-run mode — no jobs will be submitted.", debug_log_path)

    # --- experiment loop ---
    failures = []  # type: List[Tuple[str, int]]

    for index in range(1, args.num_experiments + 1):
        run_id = _generate_run_id(args.run_id_length)

        sampled = {param.path: param.sample() for param in sweep_params}

        overlay = _build_overlay(
            sweep_params,
            sampled,
            fixed_from_spec,
            args.set,
            args.tag,
            args.sweep_name,
        )
        overlay_path = config_output_dir / (
            "pretrain_overlay_%s_%02d_%s.yml" % (timestamp, index, run_id)
        )
        _write_overlay_config(overlay_path, overlay)
        overlay_arg = _launcher_config_arg(repo_root, overlay_path)

        # write CSV row
        with csv_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(
                _csv_row(
                    run_id,
                    args.sweep_name,
                    base_config_arg,
                    overlay_arg,
                    sweep_params,
                    sampled,
                    finetune_arg,
                )
            )

        # build launcher command
        if with_finetune:
            command = [
                str(launcher),
                "--run-id", run_id,
                "--chain-jobs", str(args.stage1_jobs), str(args.stage2_jobs),
                "--base-config", base_config_arg,
                "--config", overlay_arg, finetune_arg,
                "--nodes", str(args.nodes),
            ]
        else:
            command = [
                str(launcher),
                "--run-id", run_id,
                "--base-config", base_config_arg,
                "--config", overlay_arg,
                "--nodes", str(args.nodes),
            ]
        if args.account is not None:
            command.extend(["--account", args.account])
        command.extend(launcher_args)

        # log experiment details
        _log("--- Experiment %d/%d [%s] ---" % (index, args.num_experiments, run_id), debug_log_path)
        for param in sweep_params:
            value = sampled[param.path]
            formatted = repr(value) if isinstance(value, float) else str(value)
            _log("  %-60s = %s" % (param.path, formatted), debug_log_path)
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
                "ERROR: submission failed for %s with exit code %d" % (run_id, result.returncode),
                debug_log_path,
            )
        else:
            stage_tag = "-stage1" if with_finetune else ""
            _log(
                "Submitted successfully. Expected output: output/%s%s" % (run_id, stage_tag),
                debug_log_path,
            )

    _log("=== Sweep complete ===", debug_log_path)
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
