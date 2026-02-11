#!/usr/bin/env python3
"""
Generate publication-quality figures from evaluation results.

Supports multi-seed results: std_return in comparison_table.csv represents
cross-seed variation. Training curves overlay all seeds with mean +/- std bands.

Produces IEEE-style plots:
    - Fig 1: Grouped bar chart (return under disturbances)
    - Fig 2: Robustness ratio heatmap
    - Fig 3: Survival (episode length) comparison
    - Fig 4: Training curves (multi-seed with confidence bands)
    - Fig 5: Performance drop from nominal
    - Table 1: Main comparison (LaTeX)
    - Table 2: Robustness ratio (LaTeX)

Usage:
    python scripts/plot_results.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Style: IEEE double-column, serif, publication-ready ─────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Config ──────────────────────────────────────────────────────────
RESULTS_CSV = "results/Ant_v5/comparison_table.csv"
LOG_DIR = "logs"
OUTPUT_DIR = "results/figures"

METHODS = ["vanilla", "rarl", "sa_mdp", "dr", "rzsm"]
SCENARIOS = ["nominal", "force", "params", "noise", "combined"]

METHOD_LABELS = {
    "vanilla": "Vanilla DDPG",
    "rarl": "RARL",
    "sa_mdp": "SA-MDP",
    "dr": "Domain Rand.",
    "rzsm": "RZSM (Ours)",
}

SCENARIO_LABELS = {
    "nominal": "Nominal",
    "force": "Ext. Force",
    "params": "Param. Perturb.",
    "noise": "Obs. Noise",
    "combined": "Combined",
}

# Color palette: accessible, print-friendly
COLORS = {
    "vanilla": "#4C72B0",
    "rarl":    "#DD8452",
    "sa_mdp":  "#55A868",
    "dr":      "#C44E52",
    "rzsm":    "#8172B3",
}

HATCHES = {
    "vanilla": "",
    "rarl":    "//",
    "sa_mdp":  "\\\\",
    "dr":      "xx",
    "rzsm":    "",
}


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    return df


def _discover_seed_csvs(method: str) -> list[str]:
    """Find all training_returns.csv files across seed directories."""
    method_dir = os.path.join(LOG_DIR, method)
    csvs = []

    if not os.path.isdir(method_dir):
        return csvs

    # Multi-seed layout: logs/method/seed_*/training_returns.csv
    for name in sorted(os.listdir(method_dir)):
        if name.startswith("seed_"):
            csv_file = os.path.join(method_dir, name, "training_returns.csv")
            if os.path.isfile(csv_file):
                csvs.append(csv_file)

    # Fallback: legacy single-seed layout
    if not csvs:
        legacy = os.path.join(method_dir, "training_returns.csv")
        if os.path.isfile(legacy):
            csvs.append(legacy)

    return csvs


def fig1_grouped_bar(df):
    """Fig 1: Grouped bar chart — mean return per method per scenario."""
    fig, ax = plt.subplots(figsize=(7, 3.2))

    n_methods = len(METHODS)
    n_scenarios = len(SCENARIOS)
    x = np.arange(n_scenarios)
    width = 0.75 / n_methods

    for i, method in enumerate(METHODS):
        subset = df[df["method"] == method].set_index("scenario")
        subset = subset.reindex(SCENARIOS)
        means = subset["mean_return"].values
        stds = subset["std_return"].values

        bars = ax.bar(
            x + i * width - 0.375 + width / 2,
            means, width,
            yerr=stds,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            hatch=HATCHES[method],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
            capsize=1.5,
            error_kw={"linewidth": 0.6, "capthick": 0.6},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.set_ylabel("Mean Episode Return")

    n_seeds = df["n_seeds"].iloc[0] if "n_seeds" in df.columns else 1
    title = f"Ant-v5: Performance Under Disturbance Scenarios"
    if n_seeds > 1:
        title += f" ({n_seeds} seeds)"
    ax.set_title(title)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
        frameon=False,
    )
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_performance_comparison.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close()


def fig2_robustness_heatmap(df):
    """Fig 2: Robustness ratio heatmap (J_scenario / J_nominal)."""
    nominal = df[df["scenario"] == "nominal"].set_index("method")["mean_return"]

    disturb_scenarios = [s for s in SCENARIOS if s != "nominal"]
    ratio_data = np.zeros((len(METHODS), len(disturb_scenarios)))

    for i, method in enumerate(METHODS):
        for j, scenario in enumerate(disturb_scenarios):
            row = df[(df["method"] == method) & (df["scenario"] == scenario)]
            if len(row) > 0 and method in nominal.index and nominal[method] != 0:
                ratio_data[i, j] = row["mean_return"].values[0] / nominal[method]
            else:
                ratio_data[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(5, 2.8))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(ratio_data, cmap=cmap, vmin=0, vmax=1.5, aspect="auto")

    ax.set_xticks(range(len(disturb_scenarios)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in disturb_scenarios])
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])

    # Annotate cells
    for i in range(len(METHODS)):
        for j in range(len(disturb_scenarios)):
            val = ratio_data[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 or val > 1.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$J_{\mathrm{rob}} / J_{\mathrm{nom}}$", fontsize=9)
    ax.set_title("Robustness Ratio (closer to 1.0 = more robust)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_robustness_heatmap.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close()


def fig3_episode_length(df):
    """Fig 3: Episode length (survival) comparison."""
    fig, ax = plt.subplots(figsize=(5, 3))

    n_scenarios = len(SCENARIOS)
    x = np.arange(n_scenarios)
    width = 0.75 / len(METHODS)

    for i, method in enumerate(METHODS):
        subset = df[df["method"] == method].set_index("scenario")
        subset = subset.reindex(SCENARIOS)
        lengths = subset["mean_length"].values

        ax.bar(
            x + i * width - 0.375 + width / 2,
            lengths, width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.set_ylabel("Mean Episode Length (Survival)")
    ax.set_title("Ant-v5: Agent Survival Under Disturbances")
    ax.axhline(y=1000, color="gray", linewidth=0.8, linestyle="--",
               label="Max (1000)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_survival.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close()


def fig4_training_curves():
    """Fig 4: Training curves with multi-seed mean +/- std bands."""
    fig, ax = plt.subplots(figsize=(5.5, 3))

    for method in METHODS:
        seed_csvs = _discover_seed_csvs(method)
        if not seed_csvs:
            continue

        # Load all seeds
        all_returns = []
        common_steps = None

        for csv_file in seed_csvs:
            data = pd.read_csv(csv_file)
            steps_col = data.columns[0]
            ret_col = data.columns[1]
            steps = data[steps_col].values
            returns = data[ret_col].values

            all_returns.append(returns)
            if common_steps is None:
                common_steps = steps

        if common_steps is None:
            continue

        # Truncate to shortest seed length
        min_len = min(len(r) for r in all_returns)
        common_steps = common_steps[:min_len]
        all_returns = np.array([r[:min_len] for r in all_returns])

        # Compute cross-seed statistics
        mean_ret = np.mean(all_returns, axis=0)
        std_ret = np.std(all_returns, axis=0)

        # Smooth for readability
        window = min(5, max(1, min_len // 10))
        mean_smooth = pd.Series(mean_ret).rolling(window=window, min_periods=1).mean().values
        std_smooth = pd.Series(std_ret).rolling(window=window, min_periods=1).mean().values

        label = METHOD_LABELS[method]
        if len(seed_csvs) > 1:
            label += f" ({len(seed_csvs)} seeds)"

        ax.plot(
            common_steps, mean_smooth,
            label=label,
            color=COLORS[method],
            linewidth=1.0,
            alpha=0.9,
        )

        # Confidence band: mean +/- std across seeds
        ax.fill_between(
            common_steps,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=COLORS[method],
            alpha=0.15,
        )

    n_seeds = max(len(_discover_seed_csvs(m)) for m in METHODS
                  if _discover_seed_csvs(m))
    title = "Training Curves — Ant-v5"
    if n_seeds > 1:
        title += f" (mean ± std, {n_seeds} seeds)"

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_training_curves.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close()


def fig5_performance_drop(df):
    """Fig 5: Performance drop from nominal (waterfall-style)."""
    nominal = df[df["scenario"] == "nominal"].set_index("method")["mean_return"]

    fig, axes = plt.subplots(1, 4, figsize=(7, 2.5), sharey=True)
    disturb_scenarios = [s for s in SCENARIOS if s != "nominal"]

    for ax, scenario in zip(axes, disturb_scenarios):
        drops = []
        colors_list = []
        labels = []
        for method in METHODS:
            row = df[(df["method"] == method) & (df["scenario"] == scenario)]
            if len(row) > 0 and method in nominal.index:
                drop = row["mean_return"].values[0] - nominal[method]
                drops.append(drop)
                colors_list.append(COLORS[method])
                labels.append(METHOD_LABELS[method])

        bars = ax.barh(range(len(drops)), drops, color=colors_list, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        if ax == axes[0]:
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=9)
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.set_xlabel(r"$\Delta J$", fontsize=8)

    fig.suptitle("Performance Change from Nominal", fontsize=10, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_performance_drop.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close()


def table_latex(df):
    """Generate LaTeX tables for the paper."""
    n_seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns else 1
    seed_note = f" across {n_seeds} seeds" if n_seeds > 1 else ""

    # ── Table 1: Main comparison ──
    print("\n" + "=" * 70)
    print("  TABLE 1: Performance Comparison (paste into paper.tex)")
    print("=" * 70)

    print(r"\begin{table}[!t]")
    print(r"\caption{Mean episode return ($\pm$ std" + seed_note +
          r") across disturbance scenarios on Ant-v5.}")
    print(r"\label{tab:results}")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{l" + "c" * len(SCENARIOS) + "}")
    print(r"\toprule")

    header = "Method"
    for s in SCENARIOS:
        header += f" & {SCENARIO_LABELS[s]}"
    header += r" \\"
    print(header)
    print(r"\midrule")

    for method in METHODS:
        row_str = METHOD_LABELS[method]
        for scenario in SCENARIOS:
            cell = df[(df["method"] == method) & (df["scenario"] == scenario)]
            if len(cell) > 0:
                m = cell["mean_return"].values[0]
                s = cell["std_return"].values[0]
                row_str += f" & ${m:.0f} \\pm {s:.0f}$"
            else:
                row_str += " & --"
        row_str += r" \\"
        if method == "rzsm":
            print(r"\midrule")
        print(row_str)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ── Table 2: Robustness ratios ──
    print("\n" + "=" * 70)
    print("  TABLE 2: Robustness Ratios (paste into paper.tex)")
    print("=" * 70)

    nominal = df[df["scenario"] == "nominal"].set_index("method")["mean_return"]
    disturb = [s for s in SCENARIOS if s != "nominal"]

    print(r"\begin{table}[!t]")
    print(r"\caption{Robustness ratio $J_{\text{rob}} / J_{\text{nom}}$ (closer to 1.0 is better).}")
    print(r"\label{tab:robustness}")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{l" + "c" * len(disturb) + "}")
    print(r"\toprule")

    header = "Method"
    for s in disturb:
        header += f" & {SCENARIO_LABELS[s]}"
    header += r" \\"
    print(header)
    print(r"\midrule")

    for method in METHODS:
        row_str = METHOD_LABELS[method]
        for scenario in disturb:
            cell = df[(df["method"] == method) & (df["scenario"] == scenario)]
            if len(cell) > 0 and method in nominal.index and nominal[method] != 0:
                ratio = cell["mean_return"].values[0] / nominal[method]
                row_str += f" & {ratio:.2f}"
            else:
                row_str += " & --"
        row_str += r" \\"
        if method == "rzsm":
            print(r"\midrule")
        print(row_str)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading results...")
    df = load_data()
    n_seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns else 1
    print(f"  {len(df)} rows: {df['method'].nunique()} methods x "
          f"{df['scenario'].nunique()} scenarios"
          f" ({n_seeds} seed{'s' if n_seeds > 1 else ''})\n")

    fig1_grouped_bar(df)
    fig2_robustness_heatmap(df)
    fig3_episode_length(df)
    fig4_training_curves()
    fig5_performance_drop(df)
    table_latex(df)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
