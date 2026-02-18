#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for RZSM (AdversarialTD3Agent).

Two-phase search that mirrors the pipeline:
  Phase A — tune AdversarialTD3 core hyperparameters
             (learning rates, network size, TD3 knobs, adversary budget)
  Phase B — tune transformer detector hyperparameters
             (seq_len, d_model, nhead, layers, lr, train_interval)

Usage
-----
# Full search (Phase A then B), Ant-v5, 40 trials each
python scripts/tune_rzsm.py --env Ant-v5

# Phase A only
python scripts/tune_rzsm.py --env Ant-v5 --phase a --trials 60

# Phase B only (provide a pre-trained pi_opt)
python scripts/tune_rzsm.py --env Ant-v5 --phase b \
    --pi-opt-path logs/Ant_v5/nominal/seed_0/checkpoints --trials 40

# Resume a previous study
python scripts/tune_rzsm.py --env Ant-v5 --study-name rzsm_ant_phaseA

# Use multiple parallel workers (run in separate terminals)
python scripts/tune_rzsm.py --env Ant-v5 --study-name rzsm_ant_phaseA --n-jobs 1
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from src.agents.td3 import TD3Agent, AdversarialTD3Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_fn(env_name: str):
    def _thunk():
        return gym.make(env_name)
    return _thunk


def _env_safe(env: str) -> str:
    return env.replace("-", "_")


def _quick_eval(agent: AdversarialTD3Agent, env_name: str, n_episodes: int = 5) -> float:
    """Run a few evaluation episodes with pi_rob only (no adversary) and return mean return."""
    env = gym.make(env_name)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                act = agent.pi_rob(obs_t).cpu().numpy()
            obs, rew, term, trunc, _ = env.step(act)
            ep_ret += rew
            done = term or trunc
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# Phase A — core RZSM hyperparameter search
# ---------------------------------------------------------------------------

def objective_phase_a(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Tune the core TD3 + adversary knobs.
    Objective: mean return of pi_rob under NO disturbance after a short training run.
    Higher is better — Optuna maximises this.
    """
    # ── sample hyperparameters ──────────────────────────────────────────────
    pi_lr   = trial.suggest_float("pi_lr",  1e-4, 5e-3, log=True)
    q_lr    = trial.suggest_float("q_lr",   1e-4, 5e-3, log=True)
    gamma   = trial.suggest_float("gamma",  0.97, 0.999)
    polyak  = trial.suggest_float("polyak", 0.98, 0.999)
    batch_size     = trial.suggest_categorical("batch_size", [128, 256, 512])
    hidden_sizes   = trial.suggest_categorical("hidden_sizes", ["256,256", "512,256", "512,512", "256,256,256"])
    act_noise      = trial.suggest_float("act_noise", 0.05, 0.3)
    policy_delay   = trial.suggest_categorical("policy_delay", [1, 2])
    target_noise   = trial.suggest_float("target_noise", 0.1, 0.4)
    noise_clip     = trial.suggest_float("noise_clip", 0.3, 0.7)
    n_updates      = trial.suggest_categorical("n_updates", [50, 100, 200, 500])
    update_every   = trial.suggest_categorical("update_every", [10, 25, 50])
    disturbance_ratio = trial.suggest_float("disturbance_ratio", 0.02, 0.15)
    disturbance_prob  = trial.suggest_float("disturbance_prob",  0.1, 0.5)
    warmup_fraction   = trial.suggest_float("warmup_fraction",   0.05, 0.4)

    hidden = tuple(int(x) for x in hidden_sizes.split(","))

    log_dir = os.path.join(
        "logs", _env_safe(args.env), "tuning", "phase_a",
        f"trial_{trial.number:04d}"
    )

    try:
        agent = AdversarialTD3Agent(
            _make_env_fn(args.env),
            hidden_sizes=hidden,
            seed=args.seed,
            epochs=args.tune_epochs,
            steps_per_epoch=args.steps_per_epoch,
            replay_size=500_000,
            gamma=gamma,
            polyak=polyak,
            pi_lr=pi_lr,
            q_lr=q_lr,
            batch_size=batch_size,
            start_steps=args.start_steps,
            update_after=args.start_steps,
            update_every=update_every,
            n_updates=n_updates,
            act_noise=act_noise,
            max_ep_len=1_000,
            save_freq=999,          # don't save during tuning
            device=args.device,
            log_dir=log_dir,
            disturbance_ratio=disturbance_ratio,
            disturbance_probability=disturbance_prob,
            warmup_fraction=warmup_fraction,
            use_transformer=False,  # tune core first, detector separately
            pi_opt_path=args.pi_opt_path,
            num_envs=args.num_envs,
            policy_delay=policy_delay,
            target_noise=target_noise,
            noise_clip=noise_clip,
        )
        agent.train()
        score = _quick_eval(agent, args.env, n_episodes=args.eval_episodes)

    except Exception as e:
        print(f"  [trial {trial.number}] FAILED: {e}")
        score = float("-inf")
    finally:
        # Free GPU memory between trials
        try:
            del agent
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  [trial {trial.number}] score={score:.1f}  "
          f"pi_lr={pi_lr:.2e}  hidden={hidden_sizes}  "
          f"gamma={gamma:.4f}  dist_ratio={disturbance_ratio:.3f}")
    return score


# ---------------------------------------------------------------------------
# Phase B — transformer detector hyperparameter search
# ---------------------------------------------------------------------------

def objective_phase_b(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Tune the transformer detector knobs, using best Phase A params if available.
    Objective: mean return of the BLENDED policy (pi_opt + pi_rob via detector).
    """
    seq_len           = trial.suggest_categorical("seq_len",  [10, 20, 30, 50])
    d_model           = trial.suggest_categorical("d_model",  [64, 128, 256])
    nhead             = trial.suggest_categorical("nhead",    [2, 4, 8])
    num_layers        = trial.suggest_int("num_layers",        2, 6)
    detector_lr       = trial.suggest_float("detector_lr",    1e-5, 1e-3, log=True)
    train_interval    = trial.suggest_categorical("train_interval", [50, 100, 200, 500])
    warmup_fraction   = trial.suggest_float("warmup_fraction", 0.05, 0.4)

    # Ensure nhead divides d_model
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    # Use best Phase A params if a study exists, else fall back to defaults
    best_a = _load_best_phase_a(args)

    hidden = tuple(int(x) for x in best_a.get("hidden_sizes", "256,256").split(","))

    log_dir = os.path.join(
        "logs", _env_safe(args.env), "tuning", "phase_b",
        f"trial_{trial.number:04d}"
    )

    try:
        agent = AdversarialTD3Agent(
            _make_env_fn(args.env),
            hidden_sizes=hidden,
            seed=args.seed,
            epochs=args.tune_epochs,
            steps_per_epoch=args.steps_per_epoch,
            replay_size=500_000,
            gamma=best_a.get("gamma", 0.99),
            polyak=best_a.get("polyak", 0.995),
            pi_lr=best_a.get("pi_lr", 3e-4),
            q_lr=best_a.get("q_lr", 3e-4),
            batch_size=best_a.get("batch_size", 256),
            start_steps=args.start_steps,
            update_after=args.start_steps,
            update_every=best_a.get("update_every", 50),
            n_updates=best_a.get("n_updates", 200),
            act_noise=best_a.get("act_noise", 0.1),
            max_ep_len=1_000,
            save_freq=999,
            device=args.device,
            log_dir=log_dir,
            disturbance_ratio=best_a.get("disturbance_ratio", 0.05),
            disturbance_probability=best_a.get("disturbance_prob", 0.3),
            warmup_fraction=warmup_fraction,
            use_transformer=True,
            pi_opt_path=args.pi_opt_path,
            transformer_seq_len=seq_len,
            transformer_d_model=d_model,
            transformer_nhead=nhead,
            transformer_layers=num_layers,
            transformer_lr=detector_lr,
            transformer_train_interval=train_interval,
            num_envs=args.num_envs,
            policy_delay=best_a.get("policy_delay", 2),
            target_noise=best_a.get("target_noise", 0.2),
            noise_clip=best_a.get("noise_clip", 0.5),
        )
        agent.train()
        score = _quick_eval(agent, args.env, n_episodes=args.eval_episodes)

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  [trial {trial.number}] FAILED: {e}")
        score = float("-inf")
    finally:
        try:
            del agent
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  [trial {trial.number}] score={score:.1f}  "
          f"seq={seq_len}  d_model={d_model}  nhead={nhead}  "
          f"layers={num_layers}  det_lr={detector_lr:.2e}")
    return score


def _load_best_phase_a(args: argparse.Namespace) -> dict:
    """Return best Phase A params if that study exists, else empty dict."""
    study_name_a = f"rzsm_{_env_safe(args.env)}_phaseA"
    storage = f"sqlite:///logs/{_env_safe(args.env)}/tuning/optuna.db"
    try:
        study = optuna.load_study(study_name=study_name_a, storage=storage)
        return study.best_params
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def _print_results(study: optuna.Study, phase: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Phase {phase.upper()} — Top 5 Trials")
    print(f"{'='*60}")
    trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > float("-inf")],
        key=lambda t: t.value, reverse=True
    )
    for i, t in enumerate(trials[:5]):
        print(f"\n  #{i+1}  trial={t.number}  score={t.value:.2f}")
        for k, v in t.params.items():
            print(f"       {k:30s} = {v}")

    print(f"\n  BEST PARAMS (trial {study.best_trial.number}, "
          f"score={study.best_value:.2f}):")
    for k, v in study.best_params.items():
        print(f"    {k:30s} = {v}")
    print()


def _write_best_train_command(study_a: optuna.Study | None,
                               study_b: optuna.Study | None,
                               args: argparse.Namespace) -> None:
    """Print the recommended src.train command using best found params."""
    safe = _env_safe(args.env)
    params: dict = {}
    if study_a is not None:
        params.update(study_a.best_params)
    if study_b is not None:
        params.update(study_b.best_params)

    hidden = params.get("hidden_sizes", "256,256")
    pi_lr  = params.get("pi_lr",  3e-4)
    q_lr   = params.get("q_lr",   3e-4)

    print(f"\n{'='*60}")
    print("  Recommended training command with best found params:")
    print(f"{'='*60}")
    cmd = [
        f"python -m src.train",
        f"  --env {args.env}",
        f"  --mode adversarial",
        f"  --epochs 200",
        f"  --steps-per-epoch {args.steps_per_epoch}",
        f"  --hidden-sizes {hidden}",
        f"  --batch-size {params.get('batch_size', 256)}",
        f"  --pi-opt-path logs/{safe}/nominal/seed_0/checkpoints",
        f"  --disturbance-ratio {params.get('disturbance_ratio', 0.05):.4f}",
        f"  --disturbance-prob {params.get('disturbance_prob', 0.3):.3f}",
        f"  --warmup-fraction {params.get('warmup_fraction', 0.2):.3f}",
        f"  --seq-len {params.get('seq_len', 20)}",
        f"  --d-model {params.get('d_model', 128)}",
        f"  --nhead {params.get('nhead', 4)}",
        f"  --transformer-layers {params.get('num_layers', 3)}",
        f"  --detector-lr {params.get('detector_lr', 1e-4):.2e}",
        f"  --detector-train-interval {params.get('train_interval', 200)}",
        f"  --log-dir logs/{safe}/rzsm_best/seed_0",
        f"  --device {args.device}",
    ]
    print(" \\\n".join(cmd))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for RZSM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env",        type=str, default="Ant-v5")
    p.add_argument("--phase",      choices=["a", "b", "both"], default="both",
                   help="Which phase to run: a=core, b=transformer, both=a then b")
    p.add_argument("--trials",     type=int, default=40,
                   help="Number of Optuna trials per phase")
    p.add_argument("--tune-epochs",     type=int, default=30,
                   help="Training epochs per trial (shorter = faster search)")
    p.add_argument("--steps-per-epoch", type=int, default=4000)
    p.add_argument("--start-steps",     type=int, default=5000,
                   help="Random exploration steps per trial (keep small for speed)")
    p.add_argument("--eval-episodes",   type=int, default=5,
                   help="Evaluation episodes used to score each trial")
    p.add_argument("--num-envs",   type=int, default=1)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--pi-opt-path", type=str, default=None,
                   help="Path to nominal policy checkpoints (required for blending eval)")
    p.add_argument("--study-name", type=str, default=None,
                   help="Optuna study name (auto-generated if not set)")
    p.add_argument("--n-jobs",     type=int, default=1,
                   help="Parallel Optuna workers (use 1 unless you have multiple GPUs)")
    p.add_argument("--timeout",    type=float, default=None,
                   help="Stop search after this many seconds (per phase)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    safe = _env_safe(args.env)

    db_dir = Path(f"logs/{safe}/tuning")
    db_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_dir}/optuna.db"

    # Silence optuna's verbose trial logs (we print our own)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_a: optuna.Study | None = None
    study_b: optuna.Study | None = None

    # ── Phase A ─────────────────────────────────────────────────────────────
    if args.phase in ("a", "both"):
        name_a = args.study_name or f"rzsm_{safe}_phaseA"
        print(f"\n{'#'*60}")
        print(f"  Phase A — Core hyperparameters")
        print(f"  env={args.env}  trials={args.trials}  "
              f"tune_epochs={args.tune_epochs}")
        print(f"  study={name_a}")
        print(f"{'#'*60}")

        study_a = optuna.create_study(
            study_name=name_a,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study_a.optimize(
            lambda trial: objective_phase_a(trial, args),
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            gc_after_trial=True,
        )
        _print_results(study_a, "A")

    # ── Phase B ─────────────────────────────────────────────────────────────
    if args.phase in ("b", "both"):
        name_b = (args.study_name or f"rzsm_{safe}_phaseB")
        if args.phase == "b" and args.study_name:
            name_b = args.study_name
        elif args.phase == "both":
            name_b = f"rzsm_{safe}_phaseB"

        print(f"\n{'#'*60}")
        print(f"  Phase B — Transformer detector hyperparameters")
        print(f"  env={args.env}  trials={args.trials}  "
              f"tune_epochs={args.tune_epochs}")
        print(f"  study={name_b}")
        print(f"{'#'*60}")

        study_b = optuna.create_study(
            study_name=name_b,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study_b.optimize(
            lambda trial: objective_phase_b(trial, args),
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            gc_after_trial=True,
        )
        _print_results(study_b, "B")

    # ── Final recommended command ────────────────────────────────────────────
    _write_best_train_command(study_a, study_b, args)

    print(f"Optuna DB saved to: {db_dir}/optuna.db")
    print("To visualise: optuna-dashboard sqlite:///logs/{safe}/tuning/optuna.db")


if __name__ == "__main__":
    main()
