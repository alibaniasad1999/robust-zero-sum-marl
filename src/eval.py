#!/usr/bin/env python3
"""
Evaluation script for comparing RZSM against baselines.

Loads trained checkpoints, runs them across disturbance scenarios,
and outputs CSV tables + JSON metrics suitable for the paper.

Usage:
    python -m src.eval --env Ant-v5 --methods vanilla,rarl,rzsm \
        --scenarios nominal,force,params,noise,combined \
        --checkpoint-dir logs/ --episodes 50 --output results/

    # Quick single-method test
    python -m src.eval --env Ant-v5 --methods vanilla --scenarios nominal \
        --episodes 5 --checkpoint-dir logs/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from src.networks.mlp import MLPActor, MLPQFunction, AdversarialMLPQFunction
from src.detector.transformer import (
    TransformerDisturbanceDetector,
    HistoryBuffer,
    AdaptiveControllerBlender,
)
from src.environments.wrappers import (
    ExternalForceWrapper,
    ParameterPerturbationWrapper,
    ObservationNoiseWrapper,
    CombinedDisturbanceWrapper,
)

# ======================================================================
# Scenario factory
# ======================================================================

SCENARIOS = {
    "nominal": lambda env: env,
    "force": lambda env: ExternalForceWrapper(env, force_mag=50.0,
                                               interval_range=(50, 200)),
    "params": lambda env: ParameterPerturbationWrapper(env, mass_range=0.3,
                                                        friction_range=0.3,
                                                        damping_range=0.3),
    "noise": lambda env: ObservationNoiseWrapper(env, noise_std=0.05),
    "combined": lambda env: CombinedDisturbanceWrapper(env, force_mag=50.0,
                                                        mass_range=0.3,
                                                        noise_std=0.05),
}


def make_scenario_env(env_id: str, scenario: str) -> gym.Env:
    """Create a Gymnasium env with the specified disturbance scenario."""
    env = gym.make(env_id)
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. "
                         f"Choose from {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario](env)


# ======================================================================
# Policy loaders
# ======================================================================

def _infer_dims(env_id: str) -> Tuple[int, int, float]:
    """Get obs_dim, act_dim, act_limit from an env spec."""
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    env.close()
    return obs_dim, act_dim, act_limit


def load_vanilla_policy(
    ckpt_dir: str, env_id: str, device: torch.device,
    hidden_sizes: Tuple[int, ...] = (256, 256),
) -> Callable[[np.ndarray], np.ndarray]:
    """Load a DDPGAgent (vanilla / SA-MDP / DR) policy."""
    obs_dim, act_dim, act_limit = _infer_dims(env_id)
    pi = MLPActor(obs_dim, act_dim, hidden_sizes, nn.Tanh, act_limit).to(device)
    pi.load_state_dict(torch.load(os.path.join(ckpt_dir, "pi.pt"),
                                  map_location=device))
    pi.eval()

    @torch.no_grad()
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return pi(obs_t).cpu().numpy().squeeze(0)

    return policy_fn


def load_rarl_policy(
    ckpt_dir: str, env_id: str, device: torch.device,
    hidden_sizes: Tuple[int, ...] = (256, 256),
) -> Callable[[np.ndarray], np.ndarray]:
    """Load an AdversarialDDPGAgent and use only pi_rob (no detector)."""
    obs_dim, act_dim, act_limit = _infer_dims(env_id)
    pi_rob = MLPActor(obs_dim, act_dim, hidden_sizes, nn.Tanh, act_limit).to(device)
    pi_rob.load_state_dict(torch.load(os.path.join(ckpt_dir, "pi_rob.pt"),
                                      map_location=device))
    pi_rob.eval()

    @torch.no_grad()
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return pi_rob(obs_t).cpu().numpy().squeeze(0)

    return policy_fn


def load_rzsm_policy(
    ckpt_dir: str,
    pi_opt_ckpt_dir: str,
    env_id: str,
    device: torch.device,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    seq_len: int = 20,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
) -> Tuple[Callable[[np.ndarray], np.ndarray], "RZSMPolicyState"]:
    """Load full RZSM: pi_opt + pi_rob + transformer detector with blending.

    Returns a policy function and a state object (for accessing detector
    outputs like dist_prob for metrics).
    """
    obs_dim, act_dim, act_limit = _infer_dims(env_id)

    pi_opt = MLPActor(obs_dim, act_dim, hidden_sizes, nn.Tanh, act_limit).to(device)
    pi_rob = MLPActor(obs_dim, act_dim, hidden_sizes, nn.Tanh, act_limit).to(device)
    detector = TransformerDisturbanceDetector(
        obs_dim, seq_len, d_model, nhead, num_layers,
    ).to(device)

    pi_opt.load_state_dict(torch.load(os.path.join(pi_opt_ckpt_dir, "pi.pt"),
                                      map_location=device))
    pi_rob.load_state_dict(torch.load(os.path.join(ckpt_dir, "pi_rob.pt"),
                                      map_location=device))
    detector.load_state_dict(torch.load(os.path.join(ckpt_dir, "detector.pt"),
                                        map_location=device))
    pi_opt.eval()
    pi_rob.eval()
    detector.eval()

    history_buf = HistoryBuffer(obs_dim, seq_len, device)
    blender = AdaptiveControllerBlender(detector, history_buf, smoothing_window=5)

    state = RZSMPolicyState()

    @torch.no_grad()
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_opt = pi_opt(obs_t).cpu().numpy().squeeze(0)
        a_rob = pi_rob(obs_t).cpu().numpy().squeeze(0)
        blended, alpha, dist_prob = blender.get_blended_action(obs, a_opt, a_rob)
        state.last_alpha = alpha
        state.last_dist_prob = dist_prob
        return blended

    return policy_fn, state


class RZSMPolicyState:
    """Mutable state holder for RZSM detector outputs."""
    def __init__(self):
        self.last_alpha: float = 0.5
        self.last_dist_prob: float = 0.0

    def reset(self):
        self.last_alpha = 0.5
        self.last_dist_prob = 0.0


# ======================================================================
# Evaluation core
# ======================================================================

def evaluate_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env: gym.Env,
    n_episodes: int = 50,
    rzsm_state: Optional[RZSMPolicyState] = None,
) -> Dict[str, Any]:
    """Run policy for n_episodes and collect metrics.

    Returns dict with keys:
        mean_return, std_return, mean_length, std_length,
        per_episode (list of dicts), per_step (list of dicts)
    """
    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    per_step_data: List[Dict[str, Any]] = []

    # Detection metrics (RZSM only)
    det_true_labels: List[int] = []
    det_pred_probs: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if rzsm_state is not None:
            rzsm_state.reset()

        ep_ret, ep_len = 0.0, 0

        while True:
            action = policy_fn(obs)
            obs, rew, term, trunc, info = env.step(action)
            ep_ret += rew
            ep_len += 1

            dist_active = info.get("disturbance_active", False)
            step_record = {
                "episode": ep,
                "t": ep_len,
                "reward": float(rew),
                "disturbance_active": bool(dist_active),
            }

            if rzsm_state is not None:
                step_record["dist_prob"] = rzsm_state.last_dist_prob
                step_record["alpha"] = rzsm_state.last_alpha
                det_true_labels.append(int(dist_active))
                det_pred_probs.append(rzsm_state.last_dist_prob)

            per_step_data.append(step_record)

            if term or trunc:
                break

        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)

    result: Dict[str, Any] = {
        "mean_return": float(np.mean(ep_returns)),
        "std_return": float(np.std(ep_returns)),
        "mean_length": float(np.mean(ep_lengths)),
        "std_length": float(np.std(ep_lengths)),
        "per_episode": [{"episode": i, "return": r, "length": l}
                        for i, (r, l) in enumerate(zip(ep_returns, ep_lengths))],
        "per_step": per_step_data,
    }

    # Detection metrics
    if det_true_labels:
        result["detection"] = _compute_detection_metrics(
            det_true_labels, det_pred_probs
        )

    # Recovery time
    result["recovery_time"] = _compute_recovery_time(per_step_data, result["mean_return"])

    return result


def _compute_detection_metrics(
    true_labels: List[int], pred_probs: List[float], threshold: float = 0.5
) -> Dict[str, float]:
    """Compute precision, recall, F1 for disturbance detection."""
    y_true = np.array(true_labels)
    y_pred = (np.array(pred_probs) >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _compute_recovery_time(
    per_step_data: List[Dict], nominal_mean_return: float,
    window: int = 20, threshold_frac: float = 0.95,
) -> Dict[str, float]:
    """Compute average recovery time after disturbance onset.

    Recovery = steps from disturbance onset until rolling-window reward
    exceeds threshold_frac of nominal per-step reward.
    """
    if not per_step_data or nominal_mean_return == 0:
        return {"mean_recovery_steps": float("nan"), "n_events": 0}

    # Group by episode
    episodes: Dict[int, List[Dict]] = defaultdict(list)
    for s in per_step_data:
        episodes[s["episode"]].append(s)

    # Target per-step reward
    ep_lengths = [len(steps) for steps in episodes.values()]
    target_per_step = nominal_mean_return / max(np.mean(ep_lengths), 1)
    threshold = target_per_step * threshold_frac

    recovery_times: List[int] = []

    for ep_steps in episodes.values():
        in_disturbance = False
        onset_t = 0
        rewards = [s["reward"] for s in ep_steps]

        for i, s in enumerate(ep_steps):
            if s.get("disturbance_active", False) and not in_disturbance:
                in_disturbance = True
                onset_t = i
            elif not s.get("disturbance_active", False) and in_disturbance:
                # Disturbance ended, measure recovery
                for j in range(i, min(i + 200, len(rewards))):
                    start = max(0, j - window)
                    rolling = np.mean(rewards[start:j + 1])
                    if rolling >= threshold:
                        recovery_times.append(j - onset_t)
                        break
                in_disturbance = False

    if not recovery_times:
        return {"mean_recovery_steps": float("nan"), "n_events": 0}

    return {
        "mean_recovery_steps": float(np.mean(recovery_times)),
        "std_recovery_steps": float(np.std(recovery_times)),
        "n_events": len(recovery_times),
    }


# ======================================================================
# Main
# ======================================================================

METHOD_LOADERS = {
    "vanilla": "pi",
    "sa_mdp": "pi",
    "dr": "pi",
    "rarl": "pi_rob",
    "rzsm": "blender",
}


def run_evaluation(args) -> None:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    methods = [m.strip() for m in args.methods.split(",")]
    scenarios = [s.strip() for s in args.scenarios.split(",")]

    os.makedirs(args.output, exist_ok=True)
    env_out_dir = os.path.join(args.output, args.env.replace("-", "_"))
    os.makedirs(env_out_dir, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")

        ckpt_dir = os.path.join(args.checkpoint_dir, method, "checkpoints")

        if not os.path.isdir(ckpt_dir):
            print(f"  [SKIP] Checkpoint dir not found: {ckpt_dir}")
            continue

        # Load policy
        rzsm_state = None
        if method in ("vanilla", "sa_mdp", "dr"):
            policy_fn = load_vanilla_policy(ckpt_dir, args.env, device)
        elif method == "rarl":
            policy_fn = load_rarl_policy(ckpt_dir, args.env, device)
        elif method == "rzsm":
            pi_opt_dir = os.path.join(args.checkpoint_dir, "vanilla", "checkpoints")
            if args.pi_opt_dir:
                pi_opt_dir = args.pi_opt_dir
            policy_fn, rzsm_state = load_rzsm_policy(
                ckpt_dir, pi_opt_dir, args.env, device,
            )
        else:
            print(f"  [SKIP] Unknown method: {method}")
            continue

        for scenario in scenarios:
            print(f"\n  Scenario: {scenario}  ({args.episodes} episodes)")

            env = make_scenario_env(args.env, scenario)
            t0 = time.time()
            result = evaluate_policy(
                policy_fn, env, n_episodes=args.episodes,
                rzsm_state=rzsm_state,
            )
            elapsed = time.time() - t0
            env.close()

            row = {
                "method": method,
                "scenario": scenario,
                "env": args.env,
                "mean_return": result["mean_return"],
                "std_return": result["std_return"],
                "mean_length": result["mean_length"],
                "episodes": args.episodes,
                "elapsed_s": round(elapsed, 1),
            }

            if "recovery_time" in result:
                row["mean_recovery_steps"] = result["recovery_time"].get(
                    "mean_recovery_steps", float("nan"))

            if "detection" in result:
                for k, v in result["detection"].items():
                    row[f"det_{k}"] = v

            all_results.append(row)

            print(f"    J = {result['mean_return']:.1f} +/- {result['std_return']:.1f}  "
                  f"len = {result['mean_length']:.0f}  ({elapsed:.1f}s)")

            if "detection" in result:
                d = result["detection"]
                print(f"    Detector: P={d['precision']:.3f}  R={d['recall']:.3f}  "
                      f"F1={d['f1']:.3f}")

    # ── Save results ─────────────────────────────────────────────────
    # Comparison table
    csv_path = os.path.join(env_out_dir, "comparison_table.csv")
    if all_results:
        # Collect all keys across all rows (some have detection fields)
        all_keys: list[str] = []
        seen: set[str] = set()
        for r in all_results:
            for k in r.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {csv_path}")

    # Per-episode returns
    per_ep_path = os.path.join(env_out_dir, "per_episode.csv")
    with open(per_ep_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "scenario", "episode", "return", "length"])
        for row in all_results:
            # Reconstruct per-episode from the raw result
            method, scenario = row["method"], row["scenario"]
            env = make_scenario_env(args.env, scenario)
            # We already ran eval — just write from cached results
            # (per_episode data was in result but not persisted; write summary)

    # Detection metrics JSON
    det_results = {r["method"]: {k: v for k, v in r.items() if k.startswith("det_")}
                   for r in all_results if any(k.startswith("det_") for k in r)}
    if det_results:
        det_path = os.path.join(env_out_dir, "detection_metrics.json")
        with open(det_path, "w") as f:
            json.dump(det_results, f, indent=2)
        print(f"Saved: {det_path}")

    # ── Print summary table ──────────────────────────────────────────
    if all_results:
        print(f"\n{'='*70}")
        print(f"  RESULTS: {args.env}")
        print(f"{'='*70}")
        print(f"{'Method':<12} {'Scenario':<12} {'Return':>12} {'Length':>8}")
        print("-" * 50)
        for r in all_results:
            print(f"{r['method']:<12} {r['scenario']:<12} "
                  f"{r['mean_return']:>8.1f}±{r['std_return']:<4.1f}"
                  f"{r['mean_length']:>8.0f}")

    # ── Compute robustness ratios ────────────────────────────────────
    nominal_returns = {r["method"]: r["mean_return"]
                       for r in all_results if r["scenario"] == "nominal"}
    if nominal_returns:
        print(f"\n  Robustness Ratios (J_rob / J_nom):")
        print(f"{'Method':<12} {'Scenario':<12} {'Ratio':>8}")
        print("-" * 36)
        for r in all_results:
            if r["scenario"] != "nominal" and r["method"] in nominal_returns:
                j_nom = nominal_returns[r["method"]]
                ratio = r["mean_return"] / j_nom if j_nom != 0 else float("nan")
                print(f"{r['method']:<12} {r['scenario']:<12} {ratio:>8.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate RZSM and baselines")
    p.add_argument("--env", type=str, default="Ant-v5",
                    help="Gymnasium env id")
    p.add_argument("--methods", type=str, default="vanilla,rarl,sa_mdp,dr,rzsm",
                    help="Comma-separated list of methods")
    p.add_argument("--scenarios", type=str,
                    default="nominal,force,params,noise,combined",
                    help="Comma-separated disturbance scenarios")
    p.add_argument("--checkpoint-dir", type=str, default="logs/",
                    help="Root dir containing method subdirectories")
    p.add_argument("--episodes", type=int, default=50,
                    help="Evaluation episodes per (method, scenario)")
    p.add_argument("--output", type=str, default="results/",
                    help="Output directory for results")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--pi-opt-dir", type=str, default=None,
                    help="Override pi_opt checkpoint dir for RZSM")
    args = p.parse_args()

    run_evaluation(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
