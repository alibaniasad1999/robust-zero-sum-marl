#!/usr/bin/env python3
"""
RZSM Parameter Analysis & Diagnostics Report Generator.

Produces a detailed analysis log suitable for inclusion in academic papers.
Analyzes training dynamics, detector performance, and compares RZSM against
baselines to identify bottlenecks and suggest improvements.

Usage:
    python scripts/analyze_rzsm.py --env HalfCheetah-v5
    python scripts/analyze_rzsm.py --env Humanoid-v5 --log-dir logs --results-dir results
    python scripts/analyze_rzsm.py --env HalfCheetah-v5 --output analysis/

Outputs:
    analysis/<env>/rzsm_analysis_report.txt   — Full text report (paper-ready)
    analysis/<env>/sweep_analysis.csv          — Sweep config ranking (if sweep exists)
    analysis/<env>/training_dynamics.csv       — Per-epoch training metrics summary
    analysis/<env>/diagnosis_summary.json      — Machine-readable diagnosis
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def load_csv(path: str) -> List[Dict[str, str]]:
    """Load a CSV file into a list of dicts."""
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def safe_float(val: Any, default: float = float("nan")) -> float:
    """Safely convert to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_training_metrics(log_dir: str, env_safe: str, method: str,
                          seed: int) -> List[Dict[str, float]]:
    """Load training_metrics.csv for a given method/seed."""
    # Try flat structure first
    path = os.path.join(log_dir, env_safe, method, f"seed_{seed}",
                        "training_metrics.csv")
    if not os.path.isfile(path):
        # Try run_0 structure
        path = os.path.join(log_dir, env_safe, "run_0", method,
                            f"seed_{seed}", "training_metrics.csv")
    if not os.path.isfile(path):
        return []
    rows = load_csv(path)
    return [{k: safe_float(v) for k, v in row.items()} for row in rows]


def load_training_returns(log_dir: str, env_safe: str, method: str,
                          seed: int) -> List[Dict[str, float]]:
    """Load training_returns.csv for a given method/seed."""
    path = os.path.join(log_dir, env_safe, method, f"seed_{seed}",
                        "training_returns.csv")
    if not os.path.isfile(path):
        path = os.path.join(log_dir, env_safe, "run_0", method,
                            f"seed_{seed}", "training_returns.csv")
    if not os.path.isfile(path):
        return []
    return load_csv(path)


def find_seeds(log_dir: str, env_safe: str, method: str) -> List[int]:
    """Discover available seeds for a method."""
    base = os.path.join(log_dir, env_safe, method)
    if not os.path.isdir(base):
        base = os.path.join(log_dir, env_safe, "run_0", method)
    if not os.path.isdir(base):
        return []
    seeds = []
    for d in sorted(os.listdir(base)):
        if d.startswith("seed_"):
            try:
                seeds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    return seeds


def analyze_gradient_dynamics(metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """Analyze gradient ratio between pi_rob and pi_adv."""
    ratios = []
    grad_robs = []
    grad_advs = []
    for m in metrics:
        gr = m.get("grad_pi_rob", float("nan"))
        ga = m.get("grad_pi_adv", float("nan"))
        if gr == gr and ga == ga and ga > 0:  # not NaN
            ratios.append(gr / ga)
            grad_robs.append(gr)
            grad_advs.append(ga)
    if not ratios:
        return {"available": False}

    import statistics
    return {
        "available": True,
        "grad_rob_adv_ratio_mean": statistics.mean(ratios),
        "grad_rob_adv_ratio_median": statistics.median(ratios),
        "grad_rob_adv_ratio_min": min(ratios),
        "grad_rob_adv_ratio_max": max(ratios),
        "grad_rob_final": grad_robs[-1] if grad_robs else float("nan"),
        "grad_adv_final": grad_advs[-1] if grad_advs else float("nan"),
        "grad_rob_mean": statistics.mean(grad_robs),
        "grad_adv_mean": statistics.mean(grad_advs),
        "n_epochs": len(ratios),
        # Early vs late ratio
        "grad_ratio_first_quarter": statistics.mean(ratios[:max(1, len(ratios)//4)]),
        "grad_ratio_last_quarter": statistics.mean(ratios[-(max(1, len(ratios)//4)):]),
    }


def analyze_detector_training(metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """Analyze detector loss convergence."""
    det_losses = []
    det_dist_losses = []
    det_alpha_losses = []
    for m in metrics:
        dl = m.get("det_loss", float("nan"))
        dd = m.get("det_dist_loss", float("nan"))
        da = m.get("det_alpha_loss", float("nan"))
        if dl == dl:  # not NaN
            det_losses.append(dl)
        if dd == dd:
            det_dist_losses.append(dd)
        if da == da:
            det_alpha_losses.append(da)

    if not det_losses:
        return {"available": False}

    import statistics
    n = len(det_losses)
    q1 = max(1, n // 4)
    return {
        "available": True,
        "det_loss_initial": det_losses[0],
        "det_loss_final": det_losses[-1],
        "det_loss_min": min(det_losses),
        "det_loss_mean": statistics.mean(det_losses),
        "det_loss_first_quarter": statistics.mean(det_losses[:q1]),
        "det_loss_last_quarter": statistics.mean(det_losses[-q1:]),
        "det_dist_loss_final": det_dist_losses[-1] if det_dist_losses else float("nan"),
        "det_alpha_loss_final": det_alpha_losses[-1] if det_alpha_losses else float("nan"),
        "converged": det_losses[-1] < det_losses[0] * 0.1 if det_losses[0] > 0 else False,
        "n_updates": n,
    }


def analyze_q_values(metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """Analyze Q-value evolution."""
    q_vals = [m.get("q_val", float("nan")) for m in metrics
              if m.get("q_val", float("nan")) == m.get("q_val", float("nan"))]
    loss_q = [m.get("loss_q", float("nan")) for m in metrics
              if m.get("loss_q", float("nan")) == m.get("loss_q", float("nan"))]

    if not q_vals:
        return {"available": False}

    import statistics
    return {
        "available": True,
        "q_val_initial": q_vals[0],
        "q_val_final": q_vals[-1],
        "q_val_max": max(q_vals),
        "q_growth_rate": (q_vals[-1] - q_vals[0]) / max(1, len(q_vals)),
        "loss_q_final": loss_q[-1] if loss_q else float("nan"),
        "loss_q_mean": statistics.mean(loss_q) if loss_q else float("nan"),
    }


def generate_report(env: str, log_dir: str, results_dir: str,
                    output_dir: str) -> str:
    """Generate the full analysis report."""
    env_safe = env.replace("-", "_")
    lines = []

    def w(text: str = ""):
        lines.append(text)

    def section(title: str):
        w()
        w("=" * 80)
        w(f"  {title}")
        w("=" * 80)
        w()

    def subsection(title: str):
        w()
        w(f"--- {title} ---")
        w()

    # ══════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════
    w("=" * 80)
    w(f"  RZSM PARAMETER ANALYSIS & DIAGNOSTICS REPORT")
    w(f"  Environment: {env}")
    w("=" * 80)
    w()
    w("This report analyzes the Robust Zero-Sum Method (RZSM) training dynamics,")
    w("compares performance against baseline methods, diagnoses failure modes,")
    w("and provides actionable recommendations for hyperparameter tuning.")
    w()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: PERFORMANCE COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    section("1. PERFORMANCE COMPARISON: RZSM vs. BASELINES")

    # Load comparison table
    comp_path = os.path.join(results_dir, env_safe, "comparison_table.csv")
    comp_data = load_csv(comp_path)

    if not comp_data:
        w(f"  [!] No comparison results found at: {comp_path}")
        w(f"      Run evaluation first: bash scripts/train_rzsm.sh --env {env}")
    else:
        # Organize by scenario
        scenarios = ["nominal", "force", "params", "noise", "combined"]
        methods_order = ["vanilla", "rarl", "sa_mdp", "dr", "rzsm"]

        # Build lookup
        perf = {}
        for row in comp_data:
            key = (row["method"], row["scenario"])
            perf[key] = {
                "mean": safe_float(row.get("mean_return")),
                "std": safe_float(row.get("std_return")),
                "recovery": safe_float(row.get("mean_recovery_steps")),
                "det_f1": safe_float(row.get("det_f1")),
                "det_precision": safe_float(row.get("det_precision")),
                "det_recall": safe_float(row.get("det_recall")),
            }

        # Get available methods
        avail_methods = sorted(set(r["method"] for r in comp_data),
                               key=lambda m: methods_order.index(m)
                               if m in methods_order else 99)
        avail_scenarios = sorted(set(r["scenario"] for r in comp_data),
                                 key=lambda s: scenarios.index(s)
                                 if s in scenarios else 99)

        subsection("1.1 Mean Return Comparison Table")
        # Header
        hdr = f"  {'Scenario':<12}"
        for m in avail_methods:
            hdr += f"  {m:>14}"
        w(hdr)
        w("  " + "-" * (12 + 16 * len(avail_methods)))

        for scen in avail_scenarios:
            row_str = f"  {scen:<12}"
            returns = {}
            for m in avail_methods:
                p = perf.get((m, scen), {})
                mean = p.get("mean", float("nan"))
                std = p.get("std", float("nan"))
                returns[m] = mean
                if mean == mean:  # not nan
                    row_str += f"  {mean:>8.1f}±{std:<4.0f}"
                else:
                    row_str += f"  {'N/A':>14}"
            # Mark best
            valid = {m: v for m, v in returns.items() if v == v}
            if valid:
                best_m = max(valid, key=valid.get)
                if best_m == "rzsm":
                    row_str += "  ★ RZSM best"
                else:
                    rzsm_val = returns.get("rzsm", float("nan"))
                    if rzsm_val == rzsm_val:
                        gap = ((valid[best_m] - rzsm_val) / abs(valid[best_m])) * 100
                        row_str += f"  ({best_m} wins, RZSM {gap:+.1f}% gap)"
            w(row_str)

        # Rank by scenario
        subsection("1.2 Method Rankings by Scenario")
        for scen in avail_scenarios:
            ranked = []
            for m in avail_methods:
                p = perf.get((m, scen), {})
                mean = p.get("mean", float("nan"))
                if mean == mean:
                    ranked.append((m, mean))
            ranked.sort(key=lambda x: -x[1])
            rank_str = f"  {scen:<12}: "
            for i, (m, v) in enumerate(ranked):
                marker = "★" if m == "rzsm" else " "
                rank_str += f"  {i+1}.{marker}{m}({v:.0f})"
            w(rank_str)

        # Robustness gap analysis
        subsection("1.3 Robustness Gap Analysis (% Drop from Nominal)")
        w(f"  {'Method':<12}  {'Force':>10}  {'Params':>10}  {'Noise':>10}  {'Combined':>10}")
        w("  " + "-" * 56)
        for m in avail_methods:
            nom = perf.get((m, "nominal"), {}).get("mean", float("nan"))
            row_str = f"  {m:<12}"
            for scen in ["force", "params", "noise", "combined"]:
                val = perf.get((m, scen), {}).get("mean", float("nan"))
                if nom == nom and val == val and abs(nom) > 0:
                    drop = ((nom - val) / abs(nom)) * 100
                    row_str += f"  {drop:>+9.1f}%"
                else:
                    row_str += f"  {'N/A':>10}"
            w(row_str)

        w()
        w("  Interpretation: Lower % drop = more robust.")
        w("  Negative values = method performs BETTER under disturbance (rare).")

        # Recovery time
        subsection("1.4 Recovery Time (steps after force disturbance)")
        for m in avail_methods:
            rec = perf.get((m, "force"), {}).get("recovery", float("nan"))
            if rec == rec:
                w(f"  {m:<12}: {rec:.1f} steps")
            else:
                w(f"  {m:<12}: N/A (no force episodes or no recovery detected)")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: RZSM TRAINING DYNAMICS
    # ══════════════════════════════════════════════════════════════════════
    section("2. RZSM TRAINING DYNAMICS ANALYSIS")

    rzsm_seeds = find_seeds(log_dir, env_safe, "rzsm")
    if not rzsm_seeds:
        w(f"  [!] No RZSM training logs found in: {log_dir}/{env_safe}/rzsm/")
    else:
        w(f"  Seeds analyzed: {rzsm_seeds}")

        all_grad_analysis = []
        all_det_analysis = []
        all_q_analysis = []

        for seed in rzsm_seeds:
            metrics = load_training_metrics(log_dir, env_safe, "rzsm", seed)
            if not metrics:
                continue
            all_grad_analysis.append(analyze_gradient_dynamics(metrics))
            all_det_analysis.append(analyze_detector_training(metrics))
            all_q_analysis.append(analyze_q_values(metrics))

        # 2.1 Gradient Balance
        subsection("2.1 Adversary-Protagonist Gradient Balance")
        w("  The gradient ratio grad(pi_rob)/grad(pi_adv) reveals whether the")
        w("  adversary is learning effectively. A ratio >> 10 means the adversary")
        w("  gradient is negligible — it cannot learn to disturb the agent.")
        w()
        w(f"  {'Seed':>6}  {'Ratio Mean':>12}  {'Ratio Med':>12}  "
          f"{'Early Ratio':>12}  {'Late Ratio':>12}  {'|grad_rob|':>12}  {'|grad_adv|':>12}")
        w("  " + "-" * 80)
        for i, (seed, ga) in enumerate(zip(rzsm_seeds, all_grad_analysis)):
            if not ga.get("available"):
                w(f"  {seed:>6}  {'N/A':>12}")
                continue
            w(f"  {seed:>6}  {ga['grad_rob_adv_ratio_mean']:>12.1f}  "
              f"{ga['grad_rob_adv_ratio_median']:>12.1f}  "
              f"{ga['grad_ratio_first_quarter']:>12.1f}  "
              f"{ga['grad_ratio_last_quarter']:>12.1f}  "
              f"{ga['grad_rob_final']:>12.4f}  "
              f"{ga['grad_adv_final']:>12.4f}")

        # Diagnosis
        w()
        avg_ratio = [ga["grad_rob_adv_ratio_mean"] for ga in all_grad_analysis
                     if ga.get("available")]
        if avg_ratio:
            mean_ratio = sum(avg_ratio) / len(avg_ratio)
            if mean_ratio > 10:
                w(f"  ⚠ DIAGNOSIS: Gradient ratio = {mean_ratio:.1f}x")
                w("    The adversary gradient is ~{:.0f}x smaller than the protagonist.".format(mean_ratio))
                w("    This means the adversary cannot learn effective perturbations.")
                w("    → The zero-sum game is degenerate: pi_rob trains as if no adversary exists.")
                w()
                w("    RECOMMENDED FIXES:")
                w("    • Increase disturbance_ratio from 0.05 → 0.10–0.15")
                w("      (gives adversary more action budget)")
                w("    • Separate adversary learning rate: use pi_adv_lr = 3e-4 (> pi_rob_lr)")
                w("    • Gradient scaling: multiply adversary gradient by 5–10x")
                w("    • Curriculum: start with strong adversary (dist_ratio=0.15),")
                w("      anneal down to 0.05 as pi_rob improves")
            elif mean_ratio > 3:
                w(f"  ⚡ Gradient ratio = {mean_ratio:.1f}x — adversary is weaker but learning.")
                w("    Consider slight increase in disturbance_ratio for balance.")
            else:
                w(f"  ✓ Gradient ratio = {mean_ratio:.1f}x — well balanced.")

        # 2.2 Detector Loss
        subsection("2.2 Transformer Detector Convergence")
        w("  The detector must learn to distinguish nominal from disturbed states.")
        w("  Decreasing loss indicates learning; flat loss = detector not improving.")
        w()
        w(f"  {'Seed':>6}  {'Init Loss':>12}  {'Final Loss':>12}  "
          f"{'Min Loss':>12}  {'Early Avg':>12}  {'Late Avg':>12}  {'Converged?':>12}")
        w("  " + "-" * 80)
        for seed, da in zip(rzsm_seeds, all_det_analysis):
            if not da.get("available"):
                w(f"  {seed:>6}  {'N/A — no detector updates':>50}")
                continue
            conv_str = "YES" if da["converged"] else "NO"
            w(f"  {seed:>6}  {da['det_loss_initial']:>12.6f}  "
              f"{da['det_loss_final']:>12.6f}  "
              f"{da['det_loss_min']:>12.6f}  "
              f"{da['det_loss_first_quarter']:>12.6f}  "
              f"{da['det_loss_last_quarter']:>12.6f}  "
              f"{conv_str:>12}")

        # Detector diagnosis
        w()
        converged_count = sum(1 for da in all_det_analysis
                              if da.get("available") and da.get("converged"))
        total_det = sum(1 for da in all_det_analysis if da.get("available"))
        if total_det > 0:
            if converged_count == 0:
                w("  ⚠ DIAGNOSIS: Detector did NOT converge in any seed.")
                w("    The detector cannot reliably identify disturbances.")
                w()
                w("    RECOMMENDED FIXES:")
                w("    • Increase detector_lr from 1e-4 → 3e-4 or 5e-4")
                w("    • Decrease detector_train_interval from 200 → 50 or 100")
                w("    • Increase disturbance_prob from 0.3 → 0.5")
                w("      (more disturbed episodes = more training signal for detector)")
                w("    • Increase seq_len from 20 → 40")
                w("      (longer context window for temporal patterns)")
            elif converged_count < total_det:
                w(f"  ⚡ Detector converged in {converged_count}/{total_det} seeds.")
                w("    Inconsistent convergence — try more stable training:")
                w("    • Lower detector_lr slightly (e.g. 5e-5)")
                w("    • Use learning rate warmup for detector")
            else:
                w(f"  ✓ Detector converged in all {total_det} seeds.")

        # 2.3 Q-value analysis
        subsection("2.3 Q-Value and Policy Loss Evolution")
        w(f"  {'Seed':>6}  {'Q init':>10}  {'Q final':>10}  "
          f"{'Q max':>10}  {'Q growth/ep':>12}  {'Loss Q':>10}")
        w("  " + "-" * 60)
        for seed, qa in zip(rzsm_seeds, all_q_analysis):
            if not qa.get("available"):
                w(f"  {seed:>6}  {'N/A':>10}")
                continue
            w(f"  {seed:>6}  {qa['q_val_initial']:>10.1f}  "
              f"{qa['q_val_final']:>10.1f}  "
              f"{qa['q_val_max']:>10.1f}  "
              f"{qa['q_growth_rate']:>12.2f}  "
              f"{qa['loss_q_final']:>10.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: DETECTOR PERFORMANCE AT EVAL TIME
    # ══════════════════════════════════════════════════════════════════════
    section("3. DISTURBANCE DETECTOR EVALUATION METRICS")

    if comp_data:
        rzsm_rows = [r for r in comp_data if r["method"] == "rzsm"]
        if rzsm_rows:
            w("  The detector's ability to identify disturbances at evaluation time")
            w("  directly impacts the blending quality (alpha). Low recall → alpha≈1")
            w("  → RZSM degrades to pure pi_opt (=vanilla). Low precision → false")
            w("  alarms → unnecessary pi_rob usage → performance loss on nominal.")
            w()
            w(f"  {'Scenario':<12}  {'Precision':>10}  {'Recall':>10}  "
              f"{'F1':>10}  {'Accuracy':>10}  {'TP':>8}  {'FP':>8}  {'FN':>8}  {'TN':>8}")
            w("  " + "-" * 90)
            for row in rzsm_rows:
                scen = row["scenario"]
                prec = safe_float(row.get("det_precision"))
                rec = safe_float(row.get("det_recall"))
                f1 = safe_float(row.get("det_f1"))
                acc = safe_float(row.get("det_accuracy"))
                tp = safe_float(row.get("det_tp"))
                fp = safe_float(row.get("det_fp"))
                fn = safe_float(row.get("det_fn"))
                tn = safe_float(row.get("det_tn"))
                w(f"  {scen:<12}  {prec:>10.3f}  {rec:>10.3f}  "
                  f"{f1:>10.3f}  {acc:>10.3f}  "
                  f"{tp:>8.1f}  {fp:>8.1f}  {fn:>8.1f}  {tn:>8.1f}")

            w()
            # Analyze patterns
            nom_row = next((r for r in rzsm_rows if r["scenario"] == "nominal"), None)
            force_row = next((r for r in rzsm_rows if r["scenario"] == "force"), None)
            comb_row = next((r for r in rzsm_rows if r["scenario"] == "combined"), None)

            if nom_row:
                nom_f1 = safe_float(nom_row.get("det_f1"))
                nom_prec = safe_float(nom_row.get("det_precision"))
                nom_tn = safe_float(nom_row.get("det_tn"))
                nom_fp = safe_float(nom_row.get("det_fp"))
                if nom_f1 == 0 and nom_fp == nom_fp and nom_fp == 0:
                    w("  ✓ Nominal: No false positives — detector correctly stays silent.")
                elif nom_fp > 0:
                    fpr = nom_fp / (nom_fp + nom_tn) * 100 if (nom_fp + nom_tn) > 0 else 0
                    w(f"  ⚠ Nominal: {fpr:.1f}% false positive rate — detector triggers without disturbance!")

            if force_row:
                force_prec = safe_float(force_row.get("det_precision"))
                force_rec = safe_float(force_row.get("det_recall"))
                if force_prec < 0.1:
                    w(f"  ⚠ Force: Precision={force_prec:.3f} — nearly all detections are false positives.")
                    w("    The detector confuses normal dynamics with force disturbance.")
                if force_rec < 0.5:
                    w(f"  ⚠ Force: Recall={force_rec:.3f} — misses {(1-force_rec)*100:.0f}% of disturbances.")
                    w("    → alpha stays close to 1, RZSM ≈ vanilla under force.")

            if comb_row:
                comb_rec = safe_float(comb_row.get("det_recall"))
                comb_prec = safe_float(comb_row.get("det_precision"))
                if comb_prec > 0.9 and comb_rec < 0.5:
                    w(f"  ⚡ Combined: High precision ({comb_prec:.2f}) but low recall ({comb_rec:.2f}).")
                    w("    Detector is conservative — only fires when very confident.")
                    w("    → Increase detector sensitivity or lower detection threshold.")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: SWEEP RESULTS ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    section("4. HYPERPARAMETER SWEEP ANALYSIS")

    sweep_csv_path = os.path.join(log_dir, env_safe, "sweep_rzsm",
                                  "sweep_results.csv")
    sweep_data = load_csv(sweep_csv_path)

    if not sweep_data:
        w(f"  [!] No sweep results found at: {sweep_csv_path}")
        w(f"      Run sweep: bash scripts/sweep_rzsm.sh --env {env}")
        w()
        w("  Without sweep data, we recommend the following initial sweep axes:")
        w()
        w("  Priority 1 — Adversary strength (most impactful):")
        w("    disturbance_ratio ∈ {0.03, 0.05, 0.08, 0.10, 0.15}")
        w("    disturbance_prob  ∈ {0.3, 0.5, 0.7}")
        w()
        w("  Priority 2 — Detector training (second most impactful):")
        w("    detector_lr              ∈ {1e-4, 3e-4, 5e-4}")
        w("    detector_train_interval  ∈ {50, 100, 200}")
        w()
        w("  Priority 3 — Detector architecture:")
        w("    seq_len  ∈ {10, 20, 40}")
        w("    d_model  ∈ {64, 128, 256}")
        w("    layers   ∈ {2, 3, 4}")
        w()
        w("  Priority 4 — Network size (only if above are exhausted):")
        w("    hidden_sizes ∈ {(256,256), (400,300), (512,512)}")
    else:
        w(f"  Sweep results loaded: {len(sweep_data)} configurations")
        w()

        # Parse and rank
        configs = []
        for row in sweep_data:
            configs.append({
                "config": row.get("config", "?"),
                "hidden_sizes": row.get("hidden_sizes", "?"),
                "seq_len": row.get("seq_len", "?"),
                "d_model": row.get("d_model", "?"),
                "nhead": row.get("nhead", "?"),
                "layers": row.get("layers", "?"),
                "det_lr": row.get("det_lr", "?"),
                "det_interval": row.get("det_interval", "?"),
                "dist_ratio": row.get("dist_ratio", "?"),
                "dist_prob": row.get("dist_prob", "?"),
                "eval_nominal": safe_float(row.get("eval_nominal")),
                "eval_force": safe_float(row.get("eval_force")),
                "eval_combined": safe_float(row.get("eval_combined")),
                "final_return": safe_float(row.get("final_return")),
            })

        # Sort by eval_nominal
        configs.sort(key=lambda c: -c["eval_nominal"] if c["eval_nominal"] == c["eval_nominal"] else float("-inf"))

        subsection("4.1 Configuration Ranking (by eval_nominal)")
        w(f"  {'Rank':>4}  {'Config':<20}  {'Nominal':>10}  {'Force':>10}  "
          f"{'Combined':>10}  {'Train Ret':>10}")
        w("  " + "-" * 70)
        for i, c in enumerate(configs):
            w(f"  {i+1:>4}  {c['config']:<20}  "
              f"{c['eval_nominal']:>10.1f}  "
              f"{c['eval_force']:>10.1f}  "
              f"{c['eval_combined']:>10.1f}  "
              f"{c['final_return']:>10.1f}")

        # Sensitivity analysis
        subsection("4.2 Hyperparameter Sensitivity Analysis")

        # Group by varying parameter
        param_groups = defaultdict(list)
        for c in configs:
            param_groups[("dist_ratio", c["dist_ratio"])].append(c)
            param_groups[("d_model", c["d_model"])].append(c)
            param_groups[("seq_len", c["seq_len"])].append(c)
            param_groups[("det_lr", c["det_lr"])].append(c)
            param_groups[("hidden_sizes", c["hidden_sizes"])].append(c)

        w("  Impact of each hyperparameter on eval_nominal:")
        w()
        for param_name in ["dist_ratio", "d_model", "seq_len", "det_lr",
                           "hidden_sizes", "det_interval"]:
            values = {}
            for (pn, pv), cfgs in param_groups.items():
                if pn == param_name:
                    vals = [c["eval_nominal"] for c in cfgs
                            if c["eval_nominal"] == c["eval_nominal"]]
                    if vals:
                        values[pv] = sum(vals) / len(vals)
            if len(values) > 1:
                sorted_vals = sorted(values.items(),
                                     key=lambda x: -x[1])
                best_v, best_r = sorted_vals[0]
                worst_v, worst_r = sorted_vals[-1]
                impact = abs(best_r - worst_r)
                w(f"  {param_name:<20}: impact={impact:.1f}")
                for v, r in sorted_vals:
                    marker = " ★" if v == best_v else ""
                    w(f"    {str(v):<12} → avg nominal={r:.1f}{marker}")
                w()

        # Best config details
        if configs:
            best = configs[0]
            subsection("4.3 Best Configuration Details")
            w(f"  Config name:       {best['config']}")
            w(f"  hidden_sizes:      {best['hidden_sizes']}")
            w(f"  seq_len:           {best['seq_len']}")
            w(f"  d_model:           {best['d_model']}")
            w(f"  nhead:             {best['nhead']}")
            w(f"  layers:            {best['layers']}")
            w(f"  det_lr:            {best['det_lr']}")
            w(f"  det_interval:      {best['det_interval']}")
            w(f"  dist_ratio:        {best['dist_ratio']}")
            w(f"  dist_prob:         {best['dist_prob']}")
            w(f"  eval_nominal:      {best['eval_nominal']:.1f}")
            w(f"  eval_force:        {best['eval_force']:.1f}")
            w(f"  eval_combined:     {best['eval_combined']:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: SEED VARIANCE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    section("5. SEED VARIANCE ANALYSIS")

    per_seed_path = os.path.join(results_dir, env_safe, "per_seed_results.csv")
    per_seed_data = load_csv(per_seed_path)

    if not per_seed_data:
        w(f"  [!] No per-seed results at: {per_seed_path}")
    else:
        subsection("5.1 Per-Seed Returns (RZSM vs. Baselines)")
        rzsm_seeds_eval = [r for r in per_seed_data if r["method"] == "rzsm"]
        for scen in ["nominal", "force", "combined"]:
            w(f"  Scenario: {scen}")
            for m in ["vanilla", "rarl", "sa_mdp", "dr", "rzsm"]:
                seeds_data = [r for r in per_seed_data
                              if r["method"] == m and r["scenario"] == scen]
                if not seeds_data:
                    continue
                returns = [safe_float(r["mean_return"]) for r in seeds_data]
                valid = [v for v in returns if v == v]
                if valid:
                    marker = " ←" if m == "rzsm" else ""
                    vals_str = ", ".join(f"{v:.0f}" for v in valid)
                    w(f"    {m:<10}: [{vals_str}]{marker}")
            w()

        # Identify problematic seeds
        subsection("5.2 RZSM Outlier Seed Detection")
        rzsm_nominal = [r for r in per_seed_data
                        if r["method"] == "rzsm" and r["scenario"] == "nominal"]
        if rzsm_nominal:
            returns = [(r["seed"], safe_float(r["mean_return"]))
                       for r in rzsm_nominal]
            valid = [(s, v) for s, v in returns if v == v]
            if len(valid) >= 3:
                import statistics
                vals = [v for _, v in valid]
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0
                w(f"  RZSM nominal returns: mean={mean:.1f}, std={std:.1f}")
                w(f"  Coefficient of variation: {std/abs(mean)*100:.1f}%")
                w()
                for s, v in valid:
                    if std > 0 and abs(v - mean) > 1.5 * std:
                        w(f"  ⚠ {s}: return={v:.1f} — outlier ({(v-mean)/std:+.1f}σ)")
                    else:
                        w(f"  ✓ {s}: return={v:.1f}")
                w()
                if std / abs(mean) > 0.3:
                    w("  ⚠ DIAGNOSIS: High variance across seeds (CV > 30%).")
                    w("    This indicates unstable training — RZSM may be sensitive to")
                    w("    random initialization. Recommendations:")
                    w("    • Use more seeds (8-10) for reliable evaluation")
                    w("    • Try gradient clipping to stabilize training")
                    w("    • Consider warmup: train pi_rob for N epochs without adversary,")
                    w("      then enable adversary")
                else:
                    w("  ✓ Reasonable variance across seeds.")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6: ROOT CAUSE DIAGNOSIS & RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════
    section("6. ROOT CAUSE DIAGNOSIS & RECOMMENDATIONS")

    diagnosis = {
        "issues": [],
        "recommendations": [],
        "priority_order": [],
    }

    # Check RZSM vs baselines
    if comp_data:
        rzsm_nom = perf.get(("rzsm", "nominal"), {}).get("mean", float("nan"))
        vanilla_nom = perf.get(("vanilla", "nominal"), {}).get("mean", float("nan"))
        rarl_nom = perf.get(("rarl", "nominal"), {}).get("mean", float("nan"))

        rzsm_force = perf.get(("rzsm", "force"), {}).get("mean", float("nan"))
        rarl_force = perf.get(("rarl", "force"), {}).get("mean", float("nan"))
        vanilla_force = perf.get(("vanilla", "force"), {}).get("mean", float("nan"))

        rzsm_comb = perf.get(("rzsm", "combined"), {}).get("mean", float("nan"))
        best_comb = max(
            perf.get((m, "combined"), {}).get("mean", float("-inf"))
            for m in ["vanilla", "rarl", "sa_mdp", "dr"]
        )

        # Issue 1: RZSM worse than vanilla on nominal
        if rzsm_nom == rzsm_nom and vanilla_nom == vanilla_nom:
            if rzsm_nom < vanilla_nom * 0.9:
                gap = (vanilla_nom - rzsm_nom) / abs(vanilla_nom) * 100
                diagnosis["issues"].append(
                    f"RZSM nominal return is {gap:.1f}% below vanilla")
                diagnosis["recommendations"].append(
                    "The adversary is hurting nominal performance. Reduce "
                    "disturbance_ratio or add curriculum training.")

        # Issue 2: RZSM not better than RARL on force
        if (rzsm_force == rzsm_force and rarl_force == rarl_force
                and rzsm_force < rarl_force):
            diagnosis["issues"].append(
                f"RZSM ({rzsm_force:.0f}) loses to RARL ({rarl_force:.0f}) "
                f"on force disturbance")
            diagnosis["recommendations"].append(
                "The detector-based blending is not adding value over pure "
                "robust policy. Improve detector recall or tune alpha threshold.")

        # Issue 3: RZSM worst overall
        all_combined = []
        for m in ["vanilla", "rarl", "sa_mdp", "dr", "rzsm"]:
            v = perf.get((m, "combined"), {}).get("mean", float("nan"))
            if v == v:
                all_combined.append((m, v))
        all_combined.sort(key=lambda x: -x[1])
        if all_combined and all_combined[-1][0] == "rzsm":
            diagnosis["issues"].append(
                "RZSM ranks LAST on combined disturbance scenario")

        # Issue 4: Detector not detecting
        force_det = next((r for r in comp_data
                          if r["method"] == "rzsm" and r["scenario"] == "force"), None)
        if force_det:
            f1 = safe_float(force_det.get("det_f1"))
            if f1 < 0.1:
                diagnosis["issues"].append(
                    f"Detector F1={f1:.3f} on force — nearly blind to disturbances")
                diagnosis["recommendations"].append(
                    "Detector cannot identify force disturbances. This is the "
                    "primary bottleneck. Fix the detector before tuning other params.")

    # Print diagnosis
    if diagnosis["issues"]:
        subsection("6.1 Identified Issues")
        for i, issue in enumerate(diagnosis["issues"], 1):
            w(f"  [{i}] {issue}")

    subsection("6.2 Prioritized Recommendations")
    w("  Based on the analysis above, here is the priority order for improvement:")
    w()

    priorities = [
        ("HIGH", "Fix adversary gradient balance",
         "The adversary learns ~10-30x slower than pi_rob. Without a meaningful\n"
         "    adversary, pi_rob degenerates to vanilla. Fixes:\n"
         "    • Increase disturbance_ratio: 0.05 → 0.10 or 0.15\n"
         "    • Add separate adversary LR: pi_adv_lr = 3x pi_rob_lr\n"
         "    • Gradient scaling: multiply adversary loss by 5-10x"),

        ("HIGH", "Improve detector recall",
         "The detector is too conservative (precision=1.0 but recall≈0.4).\n"
         "    It misses 60% of disturbances, so alpha≈1 most of the time → RZSM≈vanilla.\n"
         "    Fixes:\n"
         "    • Lower detection threshold (if using sigmoid > 0.5, try > 0.3)\n"
         "    • Increase disturbance_prob: 0.3 → 0.5 (more training signal)\n"
         "    • Train detector for more steps: detector_train_interval 200 → 50"),

        ("MEDIUM", "Reduce seed variance",
         "RZSM has high variance across seeds (e.g., seed_2 collapses).\n"
         "    This indicates training instability. Fixes:\n"
         "    • Add gradient clipping (max_grad_norm=0.5 for all networks)\n"
         "    • Use pi_rob warmup: train 50K steps before enabling adversary\n"
         "    • Learning rate warmup for first 10K steps"),

        ("MEDIUM", "Tune detector architecture",
         "Try sweep over seq_len and d_model:\n"
         "    • seq_len=40 (longer history may help temporal pattern detection)\n"
         "    • d_model=256, layers=4 (bigger detector for complex envs)"),

        ("LOW", "Network size sweep",
         "hidden_sizes has minimal impact if the above issues persist.\n"
         "    Only sweep this after fixing gradient balance and detector."),
    ]

    for level, name, detail in priorities:
        w(f"  [{level}] {name}")
        w(f"    {detail}")
        w()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7: SUGGESTED NEXT EXPERIMENTS
    # ══════════════════════════════════════════════════════════════════════
    section("7. SUGGESTED NEXT EXPERIMENTS")
    w("  Based on the analysis, run these experiments in order:")
    w()
    w("  Experiment 1: Adversary Strength Sweep (most impactful)")
    w(f"    for ratio in 0.08 0.10 0.15 0.20; do")
    w(f"      python -m src.train --env {env} --mode adversarial \\")
    w(f"        --disturbance-ratio $ratio --disturbance-prob 0.5 \\")
    w(f"        --epochs 50 --seed 0 --log-dir logs/{env_safe}/sweep_adv_$ratio")
    w(f"    done")
    w()
    w("  Experiment 2: Detector Training Frequency")
    w(f"    for interval in 25 50 100; do")
    w(f"      python -m src.train --env {env} --mode adversarial \\")
    w(f"        --detector-train-interval $interval --detector-lr 3e-4 \\")
    w(f"        --epochs 50 --seed 0 --log-dir logs/{env_safe}/sweep_det_$interval")
    w(f"    done")
    w()
    w("  Experiment 3: Disturbance Probability (detector training signal)")
    w(f"    for prob in 0.4 0.5 0.6 0.7; do")
    w(f"      python -m src.train --env {env} --mode adversarial \\")
    w(f"        --disturbance-prob $prob --disturbance-ratio 0.10 \\")
    w(f"        --epochs 50 --seed 0 --log-dir logs/{env_safe}/sweep_prob_$prob")
    w(f"    done")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 8: PAPER-READY SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    section("8. PAPER-READY SUMMARY TABLE")
    w("  Copy-paste for LaTeX/paper:")
    w()

    if comp_data:
        # LaTeX table
        w("  \\begin{table}[h]")
        w("  \\centering")
        w(f"  \\caption{{Performance comparison on {env} (mean ± std, {comp_data[0].get('n_seeds', '5')} seeds)}}")
        w("  \\begin{tabular}{l" + "c" * len(avail_scenarios) + "}")
        w("  \\toprule")
        scen_header = " & ".join(s.capitalize() for s in avail_scenarios)
        w(f"  Method & {scen_header} \\\\")
        w("  \\midrule")
        for m in avail_methods:
            cells = []
            for scen in avail_scenarios:
                p = perf.get((m, scen), {})
                mean = p.get("mean", float("nan"))
                std = p.get("std", float("nan"))
                if mean == mean:
                    # Bold if best
                    all_vals = [perf.get((mm, scen), {}).get("mean", float("-inf"))
                                for mm in avail_methods]
                    is_best = mean >= max(v for v in all_vals if v == v)
                    if is_best:
                        cells.append(f"\\textbf{{{mean:.0f}}} ± {std:.0f}")
                    else:
                        cells.append(f"{mean:.0f} ± {std:.0f}")
                else:
                    cells.append("---")
            cell_str = " & ".join(cells)
            method_name = m.upper() if m == "rzsm" else m.replace("_", "\\_")
            w(f"  {method_name} & {cell_str} \\\\")
        w("  \\bottomrule")
        w("  \\end{tabular}")
        w("  \\end{table}")

    w()
    w("=" * 80)
    w("  END OF REPORT")
    w("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="RZSM Parameter Analysis & Diagnostics Report")
    parser.add_argument("--env", type=str, required=True,
                        help="Gymnasium environment ID (e.g. HalfCheetah-v5)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Base directory for training logs")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base directory for evaluation results")
    parser.add_argument("--output", type=str, default="analysis",
                        help="Output directory for the report")
    args = parser.parse_args()

    env_safe = args.env.replace("-", "_")
    out_dir = os.path.join(args.output, env_safe)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating RZSM analysis report for {args.env}...")
    report = generate_report(args.env, args.log_dir, args.results_dir, out_dir)

    # Save report
    report_path = os.path.join(out_dir, "rzsm_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Also print to stdout
    print()
    print(report)

    # Save diagnosis JSON
    # Re-run a quick diagnosis for JSON output
    comp_path = os.path.join(args.results_dir, env_safe, "comparison_table.csv")
    comp_data = load_csv(comp_path)
    diagnosis = {"env": args.env, "issues": [], "recommendations": []}

    if comp_data:
        perf = {}
        for row in comp_data:
            key = (row["method"], row["scenario"])
            perf[key] = safe_float(row.get("mean_return"))

        rzsm_nom = perf.get(("rzsm", "nominal"), float("nan"))
        vanilla_nom = perf.get(("vanilla", "nominal"), float("nan"))
        if rzsm_nom == rzsm_nom and vanilla_nom == vanilla_nom:
            diagnosis["rzsm_nominal"] = rzsm_nom
            diagnosis["vanilla_nominal"] = vanilla_nom
            diagnosis["nominal_gap_pct"] = (
                (vanilla_nom - rzsm_nom) / abs(vanilla_nom) * 100
                if abs(vanilla_nom) > 0 else 0
            )

        # Rankings
        for scen in ["nominal", "force", "combined"]:
            ranked = []
            for m in ["vanilla", "rarl", "sa_mdp", "dr", "rzsm"]:
                v = perf.get((m, scen), float("nan"))
                if v == v:
                    ranked.append((m, v))
            ranked.sort(key=lambda x: -x[1])
            diagnosis[f"ranking_{scen}"] = [
                {"method": m, "return": v} for m, v in ranked
            ]
            rzsm_rank = next((i+1 for i, (m, _) in enumerate(ranked)
                              if m == "rzsm"), None)
            diagnosis[f"rzsm_rank_{scen}"] = rzsm_rank

    diag_path = os.path.join(out_dir, "diagnosis_summary.json")
    with open(diag_path, "w") as f:
        json.dump(diagnosis, f, indent=2, default=str)
    print(f"Diagnosis saved to: {diag_path}")


if __name__ == "__main__":
    main()
