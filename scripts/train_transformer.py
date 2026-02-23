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

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, episodes, seq_length):
        self.seq_length = seq_length
        self.episodes = episodes
        self.indices = []
        for ep_idx, ep_data in enumerate(episodes):
            T = len(ep_data[0])
            for step in range(T):
                self.indices.append((ep_idx, step))
                
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        ep_idx, step = self.indices[idx]
        obs, dist, alpha, mag = self.episodes[ep_idx]
        
        start = max(0, step - self.seq_length + 1)
        window = obs[start : step + 1]
        if window.shape[0] < self.seq_length:
            pad = np.zeros((self.seq_length - window.shape[0], obs.shape[1]), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
            
        return (
            window,
            np.array(dist[step], dtype=np.float32),
            np.array(alpha[step], dtype=np.float32),
            np.array(mag[step], dtype=np.float32)
        )


def _load_one_dir(episode_dir: str):
    """
    Load all episodes from a single episodes/ directory.

    Returns a list of tuples:
        (obs_array, dist_array, alpha_array, mag_array)
    """
    ep_files = sorted(glob.glob(os.path.join(episode_dir, "ep_*.npz")))
    if not ep_files:
        return None

    episodes = []

    for path in ep_files:
        data = np.load(path, allow_pickle=True)
        obs = data["obs"]               # (T, obs_dim)
        tags = data["disturbance_tag"]  # (T,) str array
        alphas = data["robust_weight_target"].astype(np.float32)  # (T,)
        dist_params = data["disturbance_params"].astype(np.float32)  # (T, k)

        T = obs.shape[0]
        
        dist = (tags != "none").astype(np.float32)

        # Magnitude: first column of disturbance_params is the L2 norm
        if dist_params.ndim == 2:
            mags = dist_params[:, 0]
        else:
            mags = dist_params  # (T,) already scalar per step
            
        if len(mags) < T:
            pad_len = T - len(mags)
            mags = np.concatenate([mags, np.zeros(pad_len, dtype=np.float32)])

        episodes.append((
            obs.astype(np.float32),
            dist,
            alphas,
            mags
        ))

    return episodes


def load_dataset(log_dirs: list[str]):
    """Load episodes from all dataset directories."""
    all_episodes = []

    for log_dir in log_dirs:
        episode_dir = os.path.join(log_dir, "episodes")
        if not os.path.isdir(episode_dir):
            print(f"  [warn] No episodes/ found at {log_dir}, skipping.")
            continue

        eps = _load_one_dir(episode_dir)
        if eps is None or len(eps) == 0:
            print(f"  [warn] No .npz files in {episode_dir}, skipping.")
            continue

        total_steps = sum(len(ep[0]) for ep in eps)
        dist_steps = sum(ep[1].sum() for ep in eps)
        mag_sum = sum(ep[3].sum() for ep in eps)
        
        print(f"  {log_dir}: {total_steps:,} steps  "
              f"(dist_rate={dist_steps/total_steps:.3f}  mean_mag={mag_sum/total_steps:.4f})")
              
        all_episodes.extend(eps)

    if not all_episodes:
        raise RuntimeError(
            "No episode data found in any of the provided directories.\n"
            "Run training first so that dataset/ directories are populated."
        )

    return all_episodes


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
    episodes = load_dataset(args.log_dirs)
    
    dataset = SequenceDataset(episodes, args.seq_len)
    N = len(dataset)
    print(f"\nTotal: {N:,} windows  ({time.time() - t0:.1f}s)")

    # Train / val split
    val_n = max(1, int(N * args.val_split))
    perm = np.random.permutation(N)
    val_idx  = perm[:val_n]
    train_idx = perm[val_n:]
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}")

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler, 
        num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=val_sampler, 
        num_workers=4, pin_memory=True
    )

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
        batch_losses: list[float] = []

        for obs_b, dist_b, alpha_b, mag_b in train_loader:
            obs_b = obs_b.to(device)
            dist_b = dist_b.to(device)
            alpha_b = alpha_b.to(device)
            mag_b = mag_b.to(device)
            
            m = trainer.train_step(obs_b, dist_b, alpha_b, mag_b)
            batch_losses.append(m["total_loss"])

        # Validation
        val_losses = []
        dist_losses = []
        blend_losses = []
        mag_losses = []
        
        for obs_b, dist_b, alpha_b, mag_b in val_loader:
            obs_b = obs_b.to(device)
            dist_b = dist_b.to(device)
            alpha_b = alpha_b.to(device)
            mag_b = mag_b.to(device)
            
            val_m = trainer.evaluate(obs_b, dist_b, alpha_b, mag_b)
            val_losses.append(val_m["total_loss"])
            dist_losses.append(val_m.get("loss_disturbance", 0.0))
            blend_losses.append(val_m.get("loss_blending", 0.0))
            mag_losses.append(val_m.get("loss_magnitude", 0.0))

        mean_tr  = float(np.mean(batch_losses))
        val_loss = float(np.mean(val_losses))
        
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={mean_tr:.5f}  val={val_loss:.5f}  "
            f"dist={np.mean(dist_losses):.5f}  "
            f"blend={np.mean(blend_losses):.5f}  "
            f"mag={np.mean(mag_losses):.5f}"
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
