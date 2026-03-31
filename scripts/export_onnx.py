#!/usr/bin/env python3
"""Export a Real-ESRGAN model to ONNX format.

The exported model uses dynamic axes for batch, height, and width, so it can
process images of any size at runtime.

Requirements:
    torch (already a core dependency)

Usage:
    python scripts/export_onnx.py \\
        --weights weights/realesrgan-x4.pth \\
        --output  weights/realesrgan-x4.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Real-ESRGAN to ONNX format with dynamic axes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Path to .pth weights file.")
    parser.add_argument("--output", required=True, help="Output .onnx file path.")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor.")
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=64,
        help="Spatial size of the sample input used for tracing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
    net.eval()

    sample_input = torch.zeros(
        1, 3, args.sample_size, args.sample_size, dtype=torch.float32
    )

    print(f"Exporting to ONNX (opset={args.opset}) ...")

    torch.onnx.export(
        net,
        sample_input,
        str(output_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        do_constant_folding=True,
    )

    print(f"ONNX model saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
