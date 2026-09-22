import json
from pathlib import Path
import yaml

# Keys shared between multiple pipeline steps. These are defined at the top level of the run config.
_COMMON_KEYS = (
    "dataset",
    "split",
    "mode",
    "kb",
    "note",
    "endpoint_url",
    "entity_linkers",
    "predicate_linkers",
)

# Keys that represent list objects
_LIST_KEYS = (
    "beam_limits",
    "k1_per_pass",
    "t1_per_pass",
    "k2_per_pass",
    "t2_per_pass",
    "entity_linkers",
    "predicate_linkers",
)

# Keys that represent dict objects
_DICT_KEYS = ("linker_params",)


def load_run_config(path: str) -> dict:
    """
    Loads the YAML run config specified by path.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "name" not in cfg:
        cfg["name"] = _default_name(path)
    return cfg


def _default_name(path: str) -> str:
    """
    Derive a basic run name based on the relative path of a run config.
    """
    p = Path(path).with_suffix("")
    parts = p.parts
    if "run" in parts:
        i = max(idx for idx, part in enumerate(parts) if part == "run")
        parts = parts[i + 1:]
    else:
        parts = (p.name,)
    return "_".join(parts) if parts else p.name


def _normalize(key: str, value):
    """
    Turns any YAML value into the string representation needed by argparse.
    """
    if value is None:
        return None
    if key in _LIST_KEYS:
        if isinstance(value, list):
            return ",".join(str(v) for v in value)
        return str(value)
    if key in _DICT_KEYS and isinstance(value, dict):
        return json.dumps(value)
    return value


def model_id_from_training_config(path: str) -> str:
    """
    Define the training config's output dir (or model_name_or_path if absent) as the run's specific model id.
    This is done to avoid specifying separately in training, infer and run config.
    """
    with open(path, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    ref = train_cfg.get("output_dir") or train_cfg["model_name_or_path"]
    return Path(ref).name


def apply_run_config_defaults(
    parser,
    section: str | None = None,
    config_ref_key: str | None = None,
) -> None:
    """
    Sets the parser defaults to the values defined in the run config.
    This allows args to be assigned using: explicit CLI flag > run_config value > script's own default
    """
    pre, _ = parser.parse_known_args()
    if not getattr(pre, "run_config", None):
        return

    # Get args defined by script
    cfg = load_run_config(pre.run_config)
    dest_names = {a.dest for a in parser._actions}

    # Assign shared key values to parser defaults if run config defines them
    flat = {}
    for k in _COMMON_KEYS:
        if k in cfg and k in dest_names:
            flat[k] = _normalize(k, cfg[k])

    # Assign model id from training config
    if "model_id" in dest_names and "training_config" in cfg:
        flat["model_id"] = model_id_from_training_config(cfg["training_config"])

    if "run_name" in dest_names:
        flat["run_name"] = cfg.get("run_name", cfg["name"])

    # Assigns the specified config referenced in run config to parser default for "config"
    if config_ref_key and config_ref_key in cfg and "config" in dest_names:
        flat["config"] = cfg[config_ref_key]

    # Assign args defined for the specific pipeline step
    if section:
        for k, v in (cfg.get(section) or {}).items():
            if k not in dest_names:
                raise ValueError(
                    f"run_config section '{section}' in {pre.run_config} has "
                    f"unknown key '{k}' for this script."
                )
            flat[k] = _normalize(k, v)

    parser.set_defaults(**flat)


def require(args, *names: str) -> None:
    """
    Enforce that all arguments are actually set.
    """
    missing = [n for n in names if getattr(args, n, None) in (None, "")]
    if missing:
        raise SystemExit(
            "Missing required argument(s): "
            + ", ".join(f"--{m}" for m in missing)
            + " (supply on the command line or via --run_config)."
        )


def validate_choice(args, name: str, choices) -> None:
    """
    Verify that assigned parser defaults comply with choice arguments.
    """
    val = getattr(args, name, None)
    if val is not None and val not in choices:
        raise SystemExit(f"--{name} must be one of {list(choices)}, got {val!r}")