#!/usr/bin/env python3
"""One-time wrapper over FreeToken's `ft checkpoint` conversion.

Runs `ft checkpoint --model <hf_dir> --out <ftw> --moe-backend offload`
and asserts that `quant_format == "nvfp4"` and the architecture is
`Qwen3_5MoeForCausalLM` or `Qwen3_5MoeForConditionalGeneration`.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SUPPORTED_ARCHITECTURES = {
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
}


def find_ft_binary(custom_path: str | None = None) -> str:
    if custom_path:
        p = Path(custom_path)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
        raise FileNotFoundError(f"Specified ft binary not found or not executable: {custom_path}")

    # Check FreeToken venv
    repo_root = Path(__file__).resolve().parent.parent
    venv_ft = repo_root / "FreeToken" / ".venv" / "bin" / "ft"
    if venv_ft.is_file() and os.access(venv_ft, os.X_OK):
        return str(venv_ft)

    # Check system PATH
    import shutil
    which_ft = shutil.which("ft")
    if which_ft:
        return which_ft

    raise FileNotFoundError(
        "Could not locate 'ft' binary. Either install FreeToken venv at FreeToken/.venv "
        "or specify --ft-bin."
    )


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    ld_paths = []
    if os.path.exists("/run/opengl-driver/lib"):
        ld_paths.append("/run/opengl-driver/lib")

    try:
        gcc_out = subprocess.check_output(
            ["gcc", "-print-file-name=libstdc++.so.6"], text=True
        ).strip()
        gcc_dir = str(Path(gcc_out).parent.resolve())
        if gcc_dir not in ld_paths:
            ld_paths.append(gcc_dir)
    except Exception:
        pass

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_paths.append(existing_ld)

    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    env["TRITON_LIBCUDA_PATH"] = "/run/opengl-driver/lib"
    return env


def validate_converted_ftw(out_dir: Path) -> None:
    index_path = out_dir / "freetoken_weight.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Expected FTW index missing at: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fmt = meta.get("format")
    if fmt != "freetoken_weight":
        raise ValueError(f"Expected format 'freetoken_weight', got {fmt!r}")

    quant_format = meta.get("quant_format")
    if quant_format != "nvfp4":
        raise ValueError(f"Expected quant_format 'nvfp4', got {quant_format!r}")

    config_path = out_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected config.json missing in FTW dir: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    archs = cfg.get("architectures", [])
    if not archs or not any(a in SUPPORTED_ARCHITECTURES for a in archs):
        raise ValueError(
            f"Expected architecture in {SUPPORTED_ARCHITECTURES}, got {archs}"
        )


def convert_checkpoint(
    model_dir: str,
    out_dir: str,
    moe_backend: str = "offload",
    shard_gib: int = 8,
    ft_bin: str | None = None,
) -> None:
    ft_path = find_ft_binary(ft_bin)
    env = build_env()

    cmd = [
        ft_path,
        "checkpoint",
        "--model",
        str(Path(model_dir).resolve()),
        "--out",
        str(Path(out_dir).resolve()),
        "--moe-backend",
        moe_backend,
        "--shard-gib",
        str(shard_gib),
    ]

    print(f"Running conversion: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)
    validate_converted_ftw(Path(out_dir))
    print(f"Conversion and verification successful: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HF checkpoint to FTW format and verify output."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the HuggingFace model checkpoint directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Target path for the converted FTW output directory.",
    )
    parser.add_argument(
        "--moe-backend",
        type=str,
        default="offload",
        help="MoE backend (default: offload).",
    )
    parser.add_argument(
        "--shard-gib",
        type=int,
        default=8,
        help="Shard size in GiB (default: 8).",
    )
    parser.add_argument(
        "--ft-bin",
        type=str,
        default=None,
        help="Path to the FreeToken 'ft' binary (defaults to FreeToken/.venv/bin/ft).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip conversion and only validate an existing FTW directory.",
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    if args.check_only:
        validate_converted_ftw(out_path)
        print(f"Verification successful: {out_path}")
        return

    if not args.model:
        parser.error("--model is required unless --check-only is specified.")

    convert_checkpoint(
        model_dir=args.model,
        out_dir=args.out,
        moe_backend=args.moe_backend,
        shard_gib=args.shard_gib,
        ft_bin=args.ft_bin,
    )


if __name__ == "__main__":
    main()
