#!/usr/bin/env python3
"""
Evaluation script for comparing RZSM against baselines.

Supports multi-seed aggregation: automatically discovers seed_*/checkpoints
directories and reports mean +/- std across seeds.

Usage:
    # Evaluate all seeds (auto-discovered)
    python -m src.eval --env Ant-v5 --methods vanilla,rarl,rzsm \
        --scenarios nominal,force,params,noise,combined \
        --checkpoint-dir logs/ --episodes 50 --output results/

    # Quick single-method test
    python -m src.eval --env Ant-v5 --methods vanilla --scenarios nominal \
        --episodes 5 --checkpoint-dir logs/

    # Legacy: single-seed (no seed_* subdirectories)
    python -m src.eval --env Ant-v5 --methods vanilla --scenarios nominal \
        --checkpoint-dir logs/ --episodes 5
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
# Seed discovery
# ======================================================================

def discover_seed_dirs(method_dir: str) -> List[str]:
    """Find all seed_*/checkpoints directories under a method directory.

    Falls back to method_dir/checkpoints if no seed directories exist
    (legacy single-seed layout).
    """
    seed_dirs = []
    if os.path.isdir(method_dir):
        for name in sorted(os.listdir(method_dir)):
            if name.startswith("seed_"):
                ckpt = os.path.join(method_dir, name, "checkpoints")
                if os.path.isdir(ckpt):
                    seed_dirs.append(ckpt)

    # Fallback: legacy layout without seed subdirectories
    if not seed_dirs:
        legacy = os.path.join(method_dir, "checkpoints")
        if os.path.isdir(legacy):
            seed_dirs.append(legacy)

    return seed_dirs


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
    """Load a TD3Agent (vanilla / SA-MDP / DR) policy."""
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
    """Load an AdversarialTD3Agent and use only pi_rob (no detector)."""
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
    """Load full RZSM: pi_opt + pi_rob + transformer detector with blending."""
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
    """Run policy for n_episodes and collect metrics."""
    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    per_step_data: List[Dict[str, Any]] = []

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
        "ep_returns": ep_returns,
        "ep_lengths": ep_lengths,
        "per_step": per_step_data,
    }

    if det_true_labels:
        result["detection"] = _compute_detection_metrics(
            det_true_labels, det_pred_probs
        )

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
    """Compute average recovery time after disturbance onset."""
    if not per_step_data or nominal_mean_return == 0:
        return {"mean_recovery_steps": float("nan"), "n_events": 0}

    episodes: Dict[int, List[Dict]] = defaultdict(list)
    for s in per_step_data:
        episodes[s["episode"]].append(s)

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
# Multi-seed aggregation
# ======================================================================

def aggregate_seed_results(
    seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate evaluation results across multiple seeds.

    Takes per-seed mean_return values and computes cross-seed mean and std.
    This gives the true statistical variation across training runs.
    """
    seed_means = [r["mean_return"] for r in seed_results]
    seed_lengths = [r["mean_length"] for r in seed_results]

    agg = {
        "mean_return": float(np.mean(seed_means)),
        "std_return": float(np.std(seed_means)),
        "mean_length": float(np.mean(seed_lengths)),
        "std_length": float(np.std(seed_lengths)),
        "n_seeds": len(seed_results),
        "per_seed_returns": seed_means,
        "per_seed_lengths": seed_lengths,
    }

    # Aggregate detection metrics if present
    det_results = [r["detection"] for r in seed_results if "detection" in r]
    if det_results:
        agg["detection"] = {
            k: float(np.mean([d[k] for d in det_results]))
            for k in det_results[0] if isinstance(det_results[0][k], (int, float))
        }
        agg["detection_std"] = {
            k: float(np.std([d[k] for d in det_results]))
            for k in det_results[0] if isinstance(det_results[0][k], (int, float))
        }

    # Aggregate recovery time
    rec_results = [r["recovery_time"] for r in seed_results
                   if "recovery_time" in r
                   and not np.isnan(r["recovery_time"].get("mean_recovery_steps", float("nan")))]
    if rec_results:
        agg["recovery_time"] = {
            "mean_recovery_steps": float(np.mean(
                [r["mean_recovery_steps"] for r in rec_results])),
            "std_recovery_steps": float(np.std(
                [r["mean_recovery_steps"] for r in rec_results])),
            "n_events": sum(r["n_events"] for r in rec_results),
        }
    else:
        agg["recovery_time"] = {"mean_recovery_steps": float("nan"), "n_events": 0}

    return agg


# ======================================================================
# Main
# ======================================================================

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
    per_seed_rows: List[Dict[str, Any]] = []

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")

        method_dir = os.path.join(args.checkpoint_dir, method)
        seed_dirs = discover_seed_dirs(method_dir)

        if not seed_dirs:
            print(f"  [SKIP] No checkpoints found in: {method_dir}")
            continue

        print(f"  Found {len(seed_dirs)} seed(s): "
              f"{[os.path.basename(os.path.dirname(d)) for d in seed_dirs]}")

        for scenario in scenarios:
            print(f"\n  Scenario: {scenario}  ({args.episodes} episodes/seed)")

            seed_results = []

            for si, ckpt_dir in enumerate(seed_dirs):
                seed_name = os.path.basename(os.path.dirname(ckpt_dir))

                # Load policy for this seed
                rzsm_state = None
                if method in ("vanilla", "sa_mdp", "dr"):
                    policy_fn = load_vanilla_policy(ckpt_dir, args.env, device)
                elif method == "rarl":
                    policy_fn = load_rarl_policy(ckpt_dir, args.env, device)
                elif method == "rzsm":
                    # Find matching pi_opt seed directory
                    pi_opt_base = os.path.join(args.checkpoint_dir, "vanilla")
                    if args.pi_opt_dir:
                        pi_opt_dir = args.pi_opt_dir
                    else:
                        # Try matching seed directory first, fall back to first available
                        pi_opt_seed = os.path.join(pi_opt_base, seed_name, "checkpoints")
                        if os.path.isdir(pi_opt_seed):
                            pi_opt_dir = pi_opt_seed
                        else:
                            pi_opt_dirs = discover_seed_dirs(pi_opt_base)
                            pi_opt_dir = pi_opt_dirs[0] if pi_opt_dirs else \
                                os.path.join(pi_opt_base, "checkpoints")
                    policy_fn, rzsm_state = load_rzsm_policy(
                        ckpt_dir, pi_opt_dir, args.env, device,
                    )
                else:
                    print(f"  [SKIP] Unknown method: {method}")
                    continue

                env = make_scenario_env(args.env, scenario)
                t0 = time.time()
                result = evaluate_policy(
                    policy_fn, env, n_episodes=args.episodes,
                    rzsm_state=rzsm_state,
                )
                elapsed = time.time() - t0
                env.close()

                seed_results.append(result)

                # Per-seed row for detailed CSV
                seed_row = {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed_name,
                    "env": args.env,
                    "mean_return": result["mean_return"],
                    "std_return": result["std_return"],
                    "mean_length": result["mean_length"],
                    "episodes": args.episodes,
                    "elapsed_s": round(elapsed, 1),
                }
                if "detection" in result:
                    for k, v in result["detection"].items():
                        seed_row[f"det_{k}"] = v
                per_seed_rows.append(seed_row)

                print(f"    {seed_name}: J = {result['mean_return']:.1f} "
                      f"+/- {result['std_return']:.1f}  "
                      f"len = {result['mean_length']:.0f}  ({elapsed:.1f}s)")

            # Aggregate across seeds
            if seed_results:
                agg = aggregate_seed_results(seed_results)

                row = {
                    "method": method,
                    "scenario": scenario,
                    "env": args.env,
                    "mean_return": agg["mean_return"],
                    "std_return": agg["std_return"],
                    "mean_length": agg["mean_length"],
                    "n_seeds": agg["n_seeds"],
                    "episodes": args.episodes,
                }

                if "recovery_time" in agg:
                    row["mean_recovery_steps"] = agg["recovery_time"].get(
                        "mean_recovery_steps", float("nan"))

                if "detection" in agg:
                    for k, v in agg["detection"].items():
                        row[f"det_{k}"] = v

                all_results.append(row)

                if agg["n_seeds"] > 1:
                    print(f"    >> Aggregated ({agg['n_seeds']} seeds): "
                          f"J = {agg['mean_return']:.1f} +/- {agg['std_return']:.1f}")

    # ── Save results ─────────────────────────────────────────────────
    def _write_csv(path: str, rows: List[Dict]) -> None:
        if not rows:
            return
        all_keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {path}")

    # Aggregated comparison table (main results)
    _write_csv(os.path.join(env_out_dir, "comparison_table.csv"), all_results)

    # Per-seed detailed results
    _write_csv(os.path.join(env_out_dir, "per_seed_results.csv"), per_seed_rows)

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
        print(f"  RESULTS: {args.env}  (aggregated across seeds)")
        print(f"{'='*70}")
        print(f"{'Method':<12} {'Scenario':<12} {'Return':>12} {'Length':>8} {'Seeds':>6}")
        print("-" * 56)
        for r in all_results:
            print(f"{r['method']:<12} {r['scenario']:<12} "
                  f"{r['mean_return']:>8.1f}±{r['std_return']:<4.1f}"
                  f"{r['mean_length']:>8.0f}"
                  f"{r['n_seeds']:>6}")

    # ── Robustness ratios ────────────────────────────────────────────
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
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Root dir containing method subdirectories "
                         "(default: logs/<env_safe>/)")
    p.add_argument("--episodes", type=int, default=50,
                    help="Evaluation episodes per (method, scenario, seed)")
    p.add_argument("--output", type=str, default="results/",
                    help="Output directory for results")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--pi-opt-dir", type=str, default=None,
                    help="Override pi_opt checkpoint dir for RZSM")
    args = p.parse_args()

    # Default checkpoint-dir uses env name
    if args.checkpoint_dir is None:
        env_safe = args.env.replace("-", "_")
        args.checkpoint_dir = f"logs/{env_safe}/"

    run_evaluation(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
