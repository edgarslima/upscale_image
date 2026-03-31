#!/usr/bin/env python3
"""Export a Real-ESRGAN model to a TensorRT engine (.ep).

The engine is hardware-specific: it must be regenerated when the GPU model
or the TensorRT version changes. Engines cannot be shared across hardware.

Requirements:
    pip install -r requirements/performance.txt

Usage:
    python scripts/export_tensorrt.py \\
        --weights weights/realesrgan-x4.pth \\
        --output  weights/realesrgan-x4-trt-fp16.ep \\
        --precision fp16 \\
        --min-size 64 --opt-size 512 --max-size 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Real-ESRGAN to TensorRT engine (.ep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Path to .pth weights file.")
    parser.add_argument("--output", required=True, help="Output .ep engine path.")
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Engine precision.",
    )
    parser.add_argument("--min-size", type=int, default=64, help="Min spatial size (H=W).")
    parser.add_argument("--opt-size", type=int, default=512, help="Optimal spatial size.")
    parser.add_argument("--max-size", type=int, default=2048, help="Max spatial size.")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        import torch_tensorrt
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Install TensorRT support with: pip install -r requirements/performance.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not weights_path.exists():
        print(f"Error: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {weights_path}")

    import upscale_image.models._compat  # noqa: F401
    from basicsr.archs.rrdbnet_arch import RRDBNet

    net = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=args.scale,
    )
    loadnet = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    state_dict = loadnet.get("params_ema") or loadnet.get("params") or loadnet
    net.load_state_dict(state_dict, strict=True)
    net.eval().cuda()

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    print(f"Compiling TensorRT engine — precision={args.precision} ...")

    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 3, args.min_size, args.min_size],
            opt_shape=[1, 3, args.opt_size, args.opt_size],
            max_shape=[1, 3, args.max_size, args.max_size],
            dtype=dtype,
        )
    ]
    enabled_precisions = {dtype}

    trt_model = torch_tensorrt.compile(
        net,
        inputs=inputs,
        enabled_precisions=enabled_precisions,
    )
    torch_tensorrt.save(trt_model, str(output_path), inputs=inputs)

    print(f"Engine saved to: {output_path}")
    print("Done. Re-run this script if you change GPU or TensorRT version.")


if __name__ == "__main__":
    main()
