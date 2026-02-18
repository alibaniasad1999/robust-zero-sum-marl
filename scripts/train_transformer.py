#!/usr/bin/env python3
"""
Offline transformer training script.

Aggregates episode data logged by TD3Agent and AdversarialTD3Agent from one or
more dataset directories, then trains a TransformerDisturbanceDetector on the
full combined dataset.

Each dataset directory is expected to contain an `episodes/` subdirectory with
`ep_XXXXXX.npz` files written by DatasetLogger.

Usage
-----
# Single run
python scripts/train_transformer.py \\
    --log-dirs logs/Ant_v5/nominal/seed_0/dataset \\
    --obs-dim 27 \\
    --output logs/Ant_v5/detector_offline.pt

# Aggregate across multiple envs and seeds
python scripts/train_transformer.py \\
    --log-dirs \\
        logs/Ant_v5/nominal/seed_0/dataset \\
        logs/Ant_v5/adversarial/seed_0/dataset \\
        logs/Ant_v5/samdp/seed_0/dataset \\
    --obs-dim 27 \\
    --output logs/Ant_v5/detector_offline.pt \\
    --epochs 50 --batch-size 512
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import DatasetLogger
from src.detector.transformer import (
    TransformerDisturbanceDetector,
    DisturbanceDetectorTrainer,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_one_dir(episode_dir: str, seq_length: int):
    """
    Load all windows + magnitude labels from a single episodes/ directory.

    Returns four aligned arrays:
        obs_windows  (N, seq_length, obs_dim)  float32
        dist_labels  (N,)                       float32  0/1
        alpha_labels (N,)                       float32
        mag_labels   (N,)                       float32  L2 norm of disturbance action
    """
    ep_files = sorted(glob.glob(os.path.join(episode_dir, "ep_*.npz")))
    if not ep_files:
        return None

    all_obs, all_dist, all_alpha, all_mag = [], [], [], []

    for path in ep_files:
        data = np.load(path, allow_pickle=True)
        obs = data["obs"]               # (T, obs_dim)
        tags = data["disturbance_tag"]  # (T,) str array
        alphas = data["robust_weight_target"].astype(np.float32)  # (T,)
        dist_params = data["disturbance_params"].astype(np.float32)  # (T, k)

        T = obs.shape[0]

        # Magnitude: first column of disturbance_params is the L2 norm
        if dist_params.ndim == 2:
            mags = dist_params[:, 0]
        else:
            mags = dist_params  # (T,) already scalar per step

        for i in range(T):
            # Sliding window with zero-padding at start
            start = max(0, i - seq_length + 1)
            window = obs[start: i + 1]
            if window.shape[0] < seq_length:
                pad = np.zeros(
                    (seq_length - window.shape[0], obs.shape[1]), dtype=np.float32
                )
                window = np.concatenate([pad, window], axis=0)

            all_obs.append(window.astype(np.float32))
            all_dist.append(float(tags[i] != "none"))
            all_alpha.append(float(alphas[i]))
            all_mag.append(float(mags[i]) if i < len(mags) else 0.0)

    return (
        np.array(all_obs,  dtype=np.float32),
        np.array(all_dist, dtype=np.float32),
        np.array(all_alpha, dtype=np.float32),
        np.array(all_mag,  dtype=np.float32),
    )


def load_dataset(log_dirs: list[str], seq_length: int):
    """Load and concatenate windows from all dataset directories."""
    all_obs, all_dist, all_alpha, all_mag = [], [], [], []

    for log_dir in log_dirs:
        episode_dir = os.path.join(log_dir, "episodes")
        if not os.path.isdir(episode_dir):
            print(f"  [warn] No episodes/ found at {log_dir}, skipping.")
            continue

        result = _load_one_dir(episode_dir, seq_length)
        if result is None:
            print(f"  [warn] No .npz files in {episode_dir}, skipping.")
            continue

        obs_w, dist_l, alpha_l, mag_l = result
        print(f"  {log_dir}: {len(obs_w):,} windows  "
              f"(dist_rate={dist_l.mean():.3f}  mean_mag={mag_l.mean():.4f})")
        all_obs.append(obs_w)
        all_dist.append(dist_l)
        all_alpha.append(alpha_l)
        all_mag.append(mag_l)

    if not all_obs:
        raise RuntimeError(
            "No episode data found in any of the provided directories.\n"
            "Run training first so that dataset/ directories are populated."
        )

    return (
        np.concatenate(all_obs,  axis=0),
        np.concatenate(all_dist, axis=0),
        np.concatenate(all_alpha, axis=0),
        np.concatenate(all_mag,  axis=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Offline transformer detector training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log-dirs", nargs="+", required=True,
                   help="Dataset directories (each must contain an episodes/ sub-dir)")
    p.add_argument("--obs-dim", type=int, required=True,
                   help="Observation dimension of the environment")
    p.add_argument("--output", type=str, default="logs/detector_offline.pt",
                   help="Output path for the best detector checkpoint")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction held out for validation")
    p.add_argument("--checkpoint-interval", type=int, default=5,
                   help="Save epoch checkpoint every N epochs")
    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data
    print("\nLoading dataset ...")
    t0 = time.time()
    obs, dist, alpha, mag = load_dataset(args.log_dirs, args.seq_len)
    print(f"\nTotal: {len(obs):,} windows  ({time.time() - t0:.1f}s)")
    print(f"  disturbance rate : {dist.mean():.3f}")
    print(f"  mean alpha       : {alpha.mean():.3f}")
    print(f"  mean magnitude   : {mag.mean():.4f}")

    # Train / val split
    N = len(obs)
    val_n = max(1, int(N * args.val_split))
    perm = np.random.permutation(N)
    val_idx  = perm[:val_n]
    train_idx = perm[val_n:]
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}")

    # Move to device tensors
    obs_t   = torch.from_numpy(obs).float().to(device)
    dist_t  = torch.from_numpy(dist).float().to(device)
    alpha_t = torch.from_numpy(alpha).float().to(device)
    mag_t   = torch.from_numpy(mag).float().to(device)

    # Build model
    detector = TransformerDisturbanceDetector(
        obs_dim=args.obs_dim,
        sequence_length=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.transformer_layers,
    ).to(device)

    trainer = DisturbanceDetectorTrainer(detector, learning_rate=args.lr)
    total_params = sum(param.numel() for param in detector.parameters())
    print(f"\nDetector parameters: {total_params:,}")

    # Make sure output dir exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    print()
    for epoch in range(1, args.epochs + 1):
        perm_tr = np.random.permutation(len(train_idx))
        batch_losses: list[float] = []

        for start in range(0, len(perm_tr), args.batch_size):
            end = min(start + args.batch_size, len(perm_tr))
            idx = train_idx[perm_tr[start:end]]
            m = trainer.train_step(
                obs_t[idx], dist_t[idx], alpha_t[idx], mag_t[idx]
            )
            batch_losses.append(m["total_loss"])

        # Validation
        val_m = trainer.evaluate(
            obs_t[val_idx], dist_t[val_idx], alpha_t[val_idx], mag_t[val_idx]
        )

        mean_tr  = float(np.mean(batch_losses))
        val_loss = val_m["total_loss"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={mean_tr:.5f}  val={val_loss:.5f}  "
            f"dist={val_m.get('loss_disturbance', 0):.5f}  "
            f"blend={val_m.get('loss_blending', 0):.5f}  "
            f"mag={val_m.get('loss_magnitude', 0):.5f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(detector.state_dict(), str(out_path))
            print(f"  [best saved] {out_path}")

        # Periodic checkpoint
        if epoch % args.checkpoint_interval == 0:
            ckpt = out_path.with_stem(out_path.stem + f"_ep{epoch:03d}")
            torch.save(detector.state_dict(), str(ckpt))
            print(f"  [checkpoint] {ckpt}")

    print(f"\nTraining complete.  Best val_loss={best_val_loss:.5f}")
    print(f"Best checkpoint: {out_path}")


if __name__ == "__main__":
    main()
