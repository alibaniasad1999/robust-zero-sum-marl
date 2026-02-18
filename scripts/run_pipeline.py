#!/usr/bin/env python3
"""
Full end-to-end RZSM training pipeline.

Runs every stage in the correct order for one or more environments:

  Stage 1  — SA-MDP baseline          (data collection + logging)
  Stage 2  — Domain Randomisation     (data collection + logging)
  Stage 3  — Nominal TD3              (trains pi_opt, logs dataset)
  Stage 4  — Adversarial TD3          (trains pi_rob + pi_adv, logs dataset)
  Stage 5  — Offline Transformer      (train detector on ALL logged data)
  Stage 6  — RZSM Phase 1            (fine-tune with pre-trained detector)
  Stage 7  — RZSM Phase 2            (extended fine-tune for best results)

Usage
-----
# Default: Ant-v5 with sensible defaults
python scripts/run_pipeline.py

# Choose environment(s)
python scripts/run_pipeline.py --env HalfCheetah-v5
python scripts/run_pipeline.py --env Ant-v5 HalfCheetah-v5 Humanoid-v5

# Resume: skip already-completed stages
python scripts/run_pipeline.py --env Ant-v5 --skip-baselines --skip-nominal

# Fine-tune only (stages 6-7)
python scripts/run_pipeline.py --env Ant-v5 \\
    --skip-baselines --skip-nominal --skip-adversarial --skip-transformer

# See all options
python scripts/run_pipeline.py --help
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# obs_dim lookup (extend as needed)
OBS_DIMS = {
    "Ant-v5": 27,
    "HalfCheetah-v5": 17,
    "Humanoid-v5": 376,
    "Hopper-v5": 11,
    "Walker2d-v5": 17,
    "Swimmer-v5": 8,
    "Reacher-v5": 11,
}


def _env_safe(env: str) -> str:
    return env.replace("-", "_")


def _obs_dim(env: str) -> int:
    if env in OBS_DIMS:
        return OBS_DIMS[env]
    # Try to infer at runtime
    try:
        import gymnasium as gym
        e = gym.make(env)
        dim = e.observation_space.shape[0]
        e.close()
        return dim
    except Exception:
        raise ValueError(
            f"obs_dim for '{env}' not in lookup table and could not be inferred. "
            f"Add it to OBS_DIMS in run_pipeline.py."
        )


def run(cmd: List[str], label: str, dry_run: bool = False) -> None:
    """Run a subprocess command, print output live, raise on failure."""
    print(f"\n{'='*70}")
    print(f"  [{label}]")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*70}")
    if dry_run:
        print("  (dry-run: skipped)")
        return
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{label}' failed (exit {result.returncode}).")
        sys.exit(result.returncode)
    print(f"\n  [{label}] done in {elapsed:.0f}s")


# ---------------------------------------------------------------------------
# Per-stage functions
# ---------------------------------------------------------------------------

def stage_baselines(env: str, args: argparse.Namespace, python: str) -> None:
    safe = _env_safe(env)

    # SA-MDP
    run([
        python, "-m", "src.baselines.sa_mdp",
        "--env", env,
        "--epochs", str(args.baseline_epochs),
        "--seed", str(args.seed),
        "--log-dir", f"logs/{safe}/samdp/seed_{args.seed}",
        "--device", args.device,
    ], label=f"SA-MDP [{env}]", dry_run=args.dry_run)

    # Domain Randomisation
    run([
        python, "-m", "src.baselines.domain_randomization",
        "--env", env,
        "--epochs", str(args.baseline_epochs),
        "--seed", str(args.seed),
        "--log-dir", f"logs/{safe}/domain_rand/seed_{args.seed}",
        "--device", args.device,
    ], label=f"Domain Randomisation [{env}]", dry_run=args.dry_run)


def stage_nominal(env: str, args: argparse.Namespace, python: str) -> None:
    safe = _env_safe(env)
    run([
        python, "-m", "src.train",
        "--env", env,
        "--mode", "nominal",
        "--epochs", str(args.nominal_epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--start-steps", str(args.start_steps),
        "--update-after", str(args.update_after),
        "--num-envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--log-dir", f"logs/{safe}/nominal/seed_{args.seed}",
        "--hidden-sizes", args.hidden_sizes,
    ], label=f"Nominal TD3 [{env}]", dry_run=args.dry_run)


def stage_adversarial(env: str, args: argparse.Namespace, python: str) -> None:
    safe = _env_safe(env)
    pi_opt = f"logs/{safe}/nominal/seed_{args.seed}/checkpoints"
    run([
        python, "-m", "src.train",
        "--env", env,
        "--mode", "adversarial",
        "--epochs", str(args.adversarial_epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--start-steps", str(args.start_steps),
        "--update-after", str(args.update_after),
        "--num-envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--log-dir", f"logs/{safe}/adversarial/seed_{args.seed}",
        "--hidden-sizes", args.hidden_sizes,
        "--pi-opt-path", pi_opt,
        "--disturbance-ratio", str(args.disturbance_ratio),
        "--disturbance-prob", str(args.disturbance_prob),
        "--warmup-fraction", str(args.warmup_fraction),
        "--seq-len", str(args.seq_len),
        "--d-model", str(args.d_model),
        "--nhead", str(args.nhead),
        "--transformer-layers", str(args.transformer_layers),
        "--detector-lr", str(args.detector_lr),
        "--detector-train-interval", str(args.detector_train_interval),
    ], label=f"Adversarial TD3 [{env}]", dry_run=args.dry_run)


def stage_transformer(env: str, args: argparse.Namespace, python: str) -> None:
    """Offline transformer training on ALL logged data for this env."""
    safe = _env_safe(env)
    obs_dim = _obs_dim(env)

    # Collect every dataset directory that exists
    log_dirs = []
    for mode in ["nominal", "adversarial", "samdp", "domain_rand"]:
        d = Path(f"logs/{safe}/{mode}/seed_{args.seed}/dataset")
        if d.exists():
            log_dirs.append(str(d))

    if not log_dirs:
        print(f"  [Offline Transformer] No dataset dirs found for {env}, skipping.")
        return

    output = f"logs/{safe}/detector_offline.pt"
    cmd = [
        python, "scripts/train_transformer.py",
        "--log-dirs", *log_dirs,
        "--obs-dim", str(obs_dim),
        "--output", output,
        "--epochs", str(args.transformer_epochs),
        "--batch-size", str(args.transformer_batch),
        "--seq-len", str(args.seq_len),
        "--d-model", str(args.d_model),
        "--nhead", str(args.nhead),
        "--transformer-layers", str(args.transformer_layers),
        "--lr", str(args.detector_lr),
        "--device", args.device,
    ]
    run(cmd, label=f"Offline Transformer [{env}]", dry_run=args.dry_run)


def stage_finetune(
    env: str,
    args: argparse.Namespace,
    python: str,
    phase: int,
) -> None:
    """
    RZSM fine-tuning with the offline pre-trained transformer.

    Phase 1: moderate epochs, standard disturbance — learns blending
    Phase 2: extended epochs, tighter warmup — squeezes out best performance
    """
    safe = _env_safe(env)
    pi_opt = f"logs/{safe}/nominal/seed_{args.seed}/checkpoints"

    if phase == 1:
        epochs = args.finetune_phase1_epochs
        warmup = args.warmup_fraction
        log_dir = f"logs/{safe}/rzsm_phase1/seed_{args.seed}"
        label = f"RZSM Phase 1 [{env}]"
    else:
        epochs = args.finetune_phase2_epochs
        warmup = max(0.05, args.warmup_fraction / 2)   # tighter ramp
        log_dir = f"logs/{safe}/rzsm_phase2/seed_{args.seed}"
        label = f"RZSM Phase 2 [{env}]"

    cmd = [
        python, "-m", "src.train",
        "--env", env,
        "--mode", "adversarial",
        "--epochs", str(epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--start-steps", str(args.start_steps),
        "--update-after", str(args.update_after),
        "--num-envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--log-dir", log_dir,
        "--hidden-sizes", args.hidden_sizes,
        "--pi-opt-path", pi_opt,
        "--disturbance-ratio", str(args.disturbance_ratio),
        "--disturbance-prob", str(args.disturbance_prob),
        "--warmup-fraction", str(warmup),
        "--seq-len", str(args.seq_len),
        "--d-model", str(args.d_model),
        "--nhead", str(args.nhead),
        "--transformer-layers", str(args.transformer_layers),
        "--detector-lr", str(args.detector_lr),
        "--detector-train-interval", str(args.detector_train_interval),
    ]
    run(cmd, label=label, dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full RZSM end-to-end pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- environment ----
    p.add_argument(
        "--env", nargs="+", default=["Ant-v5"],
        metavar="ENV",
        help="Gymnasium env ID(s). Multiple envs are run sequentially.",
    )

    # ---- skip flags ----
    p.add_argument("--skip-baselines",   action="store_true", help="Skip SA-MDP and Domain Rand")
    p.add_argument("--skip-nominal",     action="store_true", help="Skip nominal TD3 training")
    p.add_argument("--skip-adversarial", action="store_true", help="Skip adversarial TD3 training")
    p.add_argument("--skip-transformer", action="store_true", help="Skip offline transformer training")
    p.add_argument("--skip-finetune",    action="store_true", help="Skip RZSM fine-tuning (phases 1+2)")
    p.add_argument("--dry-run",          action="store_true", help="Print commands without running them")

    # ---- epoch counts ----
    p.add_argument("--baseline-epochs",       type=int, default=100)
    p.add_argument("--nominal-epochs",        type=int, default=200)
    p.add_argument("--adversarial-epochs",    type=int, default=200)
    p.add_argument("--transformer-epochs",    type=int, default=50)
    p.add_argument("--finetune-phase1-epochs",type=int, default=100,
                   help="RZSM Phase 1: learn blending with pre-trained detector")
    p.add_argument("--finetune-phase2-epochs",type=int, default=200,
                   help="RZSM Phase 2: extended fine-tuning for best results")

    # ---- training hyperparams (shared across stages) ----
    p.add_argument("--steps-per-epoch",  type=int,   default=4000)
    p.add_argument("--batch-size",       type=int,   default=256)
    p.add_argument("--start-steps",      type=int,   default=10000)
    p.add_argument("--update-after",     type=int,   default=10000)
    p.add_argument("--num-envs",         type=int,   default=1)
    p.add_argument("--hidden-sizes",     type=str,   default="256,256")
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--device",           type=str,   default="auto")

    # ---- adversarial hyperparams ----
    p.add_argument("--disturbance-ratio", type=float, default=0.05)
    p.add_argument("--disturbance-prob",  type=float, default=0.3)
    p.add_argument("--warmup-fraction",   type=float, default=0.2,
                   help="Epsilon annealing: ramp disturbance 0→max over this fraction of steps")

    # ---- transformer hyperparams ----
    p.add_argument("--seq-len",                 type=int,   default=20)
    p.add_argument("--d-model",                 type=int,   default=128)
    p.add_argument("--nhead",                   type=int,   default=4)
    p.add_argument("--transformer-layers",      type=int,   default=3)
    p.add_argument("--detector-lr",             type=float, default=1e-4)
    p.add_argument("--detector-train-interval", type=int,   default=200)
    p.add_argument("--transformer-batch",       type=int,   default=512)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve python executable relative to this script's venv if available
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    venv_python = repo_root / ".env" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    print(f"\n{'#'*70}")
    print(f"  RZSM Pipeline")
    print(f"  python : {python}")
    print(f"  envs   : {args.env}")
    print(f"  device : {args.device}")
    print(f"  seed   : {args.seed}")
    if args.dry_run:
        print("  *** DRY RUN — no commands will execute ***")
    print(f"{'#'*70}\n")

    pipeline_start = time.time()

    for env in args.env:
        print(f"\n{'#'*70}")
        print(f"  Environment: {env}")
        print(f"{'#'*70}")

        if not args.skip_baselines:
            stage_baselines(env, args, python)

        if not args.skip_nominal:
            stage_nominal(env, args, python)

        if not args.skip_adversarial:
            stage_adversarial(env, args, python)

        if not args.skip_transformer:
            stage_transformer(env, args, python)

        if not args.skip_finetune:
            stage_finetune(env, args, python, phase=1)
            stage_finetune(env, args, python, phase=2)

    total = time.time() - pipeline_start
    h, m = divmod(int(total), 3600)
    m, s = divmod(m, 60)
    print(f"\n{'#'*70}")
    print(f"  Pipeline complete  ({h}h {m}m {s}s)")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
