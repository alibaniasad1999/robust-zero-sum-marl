#!/usr/bin/env python3
"""Export trained PyTorch models to ONNX format for TensorRT on Jetson.

Usage:
    python3 export_onnx.py \
        --pi-opt pretrained/ant/model/pi_opt.pth \
        --pi-rob pretrained/ant/model/pi_rob.pth \
        --detector pretrained/ant/model/detector.pth \
        --obs-dim 27 --act-dim 8 --seq-len 20 \
        --output-dir models/onnx/

After export, convert to TensorRT on Jetson:
    trtexec --onnx=pi_opt.onnx --saveEngine=pi_opt.engine --fp16
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export RZSM models to ONNX for TensorRT deployment")

    p.add_argument("--pi-opt", type=str, default="",
                   help="Path to trained optimal policy .pth")
    p.add_argument("--pi-rob", type=str, default="",
                   help="Path to trained robust policy .pth")
    p.add_argument("--detector", type=str, default="",
                   help="Path to trained detector .pth")

    p.add_argument("--obs-dim", type=int, default=27,
                   help="Observation dimension (Ant=27, Humanoid=45)")
    p.add_argument("--act-dim", type=int, default=8,
                   help="Action dimension (Ant=8, Humanoid=17)")
    p.add_argument("--seq-len", type=int, default=20,
                   help="Transformer sequence length")
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256],
                   help="MLP hidden layer sizes")
    p.add_argument("--act-limit", type=float, default=1.0,
                   help="Action limit")

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dim-feedforward", type=int, default=256)

    p.add_argument("--output-dir", type=str, default="models/onnx",
                   help="Output directory for ONNX files")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version")
    return p


def export_policy(model: nn.Module, obs_dim: int, output_path: Path,
                  opset: int, name: str) -> None:
    """Export an MLPActor to ONNX."""
    model.eval()
    dummy = torch.randn(1, obs_dim)

    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"  {name}: {output_path} ({size_kb:.1f} KB)")


def export_detector(model: nn.Module, obs_dim: int, seq_len: int,
                    output_path: Path, opset: int) -> None:
    """Export TransformerDisturbanceDetector to ONNX."""
    model.eval()
    dummy = torch.randn(1, seq_len, obs_dim)

    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["obs_sequence"],
        output_names=["disturbance_prob", "blending_weight"],
        dynamic_axes={"obs_sequence": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"  detector: {output_path} ({size_kb:.1f} KB)")


def main():
    args = build_parser().parse_args()

    # Add project root to path so we can import src.*
    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root))

    from src.networks.mlp import MLPActor
    from src.detector.transformer import TransformerDisturbanceDetector

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting models (obs_dim={args.obs_dim}, act_dim={args.act_dim}):")

    # ── Export pi_opt ─────────────────────────────────────────────
    if args.pi_opt:
        pi_opt = MLPActor(
            obs_dim=args.obs_dim, act_dim=args.act_dim,
            hidden_sizes=tuple(args.hidden_sizes),
            activation=nn.Tanh, act_limit=args.act_limit)
        state = torch.load(args.pi_opt, map_location="cpu", weights_only=True)
        pi_opt.load_state_dict(state)
        export_policy(pi_opt, args.obs_dim, output_dir / "pi_opt.onnx",
                      args.opset, "pi_opt")

    # ── Export pi_rob ─────────────────────────────────────────────
    if args.pi_rob:
        pi_rob = MLPActor(
            obs_dim=args.obs_dim, act_dim=args.act_dim,
            hidden_sizes=tuple(args.hidden_sizes),
            activation=nn.Tanh, act_limit=args.act_limit)
        state = torch.load(args.pi_rob, map_location="cpu", weights_only=True)
        pi_rob.load_state_dict(state)
        export_policy(pi_rob, args.obs_dim, output_dir / "pi_rob.onnx",
                      args.opset, "pi_rob")

    # ── Export detector ───────────────────────────────────────────
    if args.detector:
        detector = TransformerDisturbanceDetector(
            obs_dim=args.obs_dim, sequence_length=args.seq_len,
            d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward)
        state = torch.load(args.detector, map_location="cpu",
                           weights_only=True)
        detector.load_state_dict(state)
        export_detector(detector, args.obs_dim, args.seq_len,
                        output_dir / "detector.onnx", args.opset)

    # ── Validate with onnx (if installed) ─────────────────────────
    try:
        import onnx
        for f in output_dir.glob("*.onnx"):
            model = onnx.load(str(f))
            onnx.checker.check_model(model)
            print(f"  Validated: {f.name}")
    except ImportError:
        print("\n  (Install 'onnx' package to validate exported models)")

    # ── Print TensorRT conversion instructions ────────────────────
    print("\nTo convert to TensorRT on Jetson:")
    for f in sorted(output_dir.glob("*.onnx")):
        engine = f.with_suffix(".engine")
        print(f"  trtexec --onnx={f} --saveEngine={engine} --fp16 "
              f"--workspace=256")


if __name__ == "__main__":
    main()
