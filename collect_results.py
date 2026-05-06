"""
Collect experiment results from Kevin, Zeph, and aleksei branches and produce:
  results/combined_results.csv   — unified per-method summary
  results/performance_bar_chart.png — bar chart for representative environments
"""

import io
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPRESENTATIVE_ENVS = [
    "Turb_TCRWP_Floris",
    "Turb1_Row1_Floris",
    "Turb2_Row1_Floris",
    "Turb3_Row1_Floris",
    "Turb6_Row2_Floris",
    "Turb16_Row5_Floris",
    "Turb32_Row5_Floris",
    "Ablaincourt_Floris",
    "HornsRev1_Floris",
    "HornsRev2_Floris",
    "Ormonde_Floris",
    "WMR_Floris",
]

METHOD_COLORS = {
    "GP":       "#4C72B0",
    "TabPFN":   "#DD8452",
    "GraphPFN": "#55A868",
    "PPO":      "#C44E52",
    "Random":   "#8172B3",
}

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git_show(branch: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"origin/{branch}:{path}"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git show origin/{branch}:{path} failed:\n{result.stderr.decode()}"
        )
    return result.stdout


def collect_stream_texts(nb: dict, cell_index: int) -> list:
    """Return all stream output text blocks from a cell."""
    return [
        "".join(out.get("text", []))
        for out in nb["cells"][cell_index].get("outputs", [])
        if out.get("output_type") == "stream"
    ]


def parse_per_seed_blocks(texts: list, scenario_label: str) -> pd.DataFrame:
    """
    Parse stream output blocks for per-seed reward rows.
    Handles two formats from Kevin's V2 notebook:
      WITH scenario col:    0  Scenario 2  Layout_Floris  seed  Method  reward
      WITHOUT scenario col: 0  Layout_Floris  seed  Method  reward
    """
    rows = []
    methods = r"(Do-Nothing|Random|GP|TabPFN|GraphPFN)"

    # Format A: has scenario column (Scenario 2 cell)
    re_a = re.compile(
        r"^\s*\d+\s+(Scenario\s+\d+)\s+(\S+_Floris)\s+(\d+)\s+"
        + methods + r"\s+([\d.]+)"
    )
    # Format B: no scenario column (Scenario 1 cell)
    re_b = re.compile(
        r"^\s*\d+\s+(\S+_Floris)\s+(\d+)\s+" + methods + r"\s+([\d.]+)"
    )

    for block in texts:
        if "_Floris" not in block:
            continue
        for line in block.splitlines():
            m = re_a.match(line)
            if m:
                rows.append({
                    "scenario": m.group(1).strip(),
                    "env_id":   m.group(2).strip(),
                    "seed":     int(m.group(3)),
                    "method":   m.group(4).strip(),
                    "reward":   float(m.group(5)),
                })
                continue
            m = re_b.match(line)
            if m:
                rows.append({
                    "scenario": scenario_label,
                    "env_id":   m.group(1).strip(),
                    "seed":     int(m.group(2)),
                    "method":   m.group(3).strip(),
                    "reward":   float(m.group(4)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kevin branch — WFCRL_GraphPFN_TabPFN_V2.ipynb
# Cell 22: Scenario 1 per-seed stream outputs
# Cell 29: Scenario 2 per-seed stream outputs
# ---------------------------------------------------------------------------

def load_kevin_results() -> pd.DataFrame:
    print("Loading Kevin branch results...")
    nb_bytes = git_show("Kevin", "CSCI_5980_notebooks/WFCRL_GraphPFN_TabPFN_V2.ipynb")
    nb = json.loads(nb_bytes)

    all_rows = []
    for cell_idx, sc_label in [(22, "Scenario 1"), (29, "Scenario 2")]:
        texts = collect_stream_texts(nb, cell_idx)
        df = parse_per_seed_blocks(texts, sc_label)
        if df.empty:
            print(f"  WARNING: no rows parsed from Kevin cell {cell_idx}")
        else:
            print(f"  {sc_label}: {len(df)} raw per-seed rows from cell {cell_idx}")
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    raw = pd.concat(all_rows, ignore_index=True).drop_duplicates()

    # Aggregate: mean/std per (scenario, env_id, method)
    agg = (
        raw.groupby(["scenario", "env_id", "method"])["reward"]
        .agg(mean_reward="mean", std_reward="std")
        .reset_index()
    )
    agg["std_reward"] = agg["std_reward"].fillna(0.0)

    # Compute gain vs Do-Nothing baseline per (scenario, env_id)
    baseline = (
        agg[agg["method"] == "Do-Nothing"]
        .rename(columns={"mean_reward": "baseline_mean"})
        [["scenario", "env_id", "baseline_mean"]]
    )
    agg = agg.merge(baseline, on=["scenario", "env_id"])
    agg["mean_gain_kw"] = agg["mean_reward"] - agg["baseline_mean"]
    agg["std_gain_kw"]  = agg["std_reward"]
    agg["gain_pct"]     = (agg["mean_gain_kw"] / agg["baseline_mean"] * 100).round(3)
    agg["branch"]       = "Kevin"

    result = agg[agg["method"] != "Do-Nothing"].copy()
    print(f"  Aggregated: {len(result)} rows, scenarios={sorted(result['scenario'].unique())}")
    return result[["branch", "scenario", "env_id", "method",
                   "mean_gain_kw", "std_gain_kw", "gain_pct",
                   "mean_reward", "std_reward", "baseline_mean"]]


# ---------------------------------------------------------------------------
# Zeph branch — RL_PPO_Scenario_{1,2}_full.ipynb
# Finds the execute_result cell with env_id/ppo_mean/baseline_mean columns
# and parses the truncated two-block repr with regex.
# ---------------------------------------------------------------------------

def _parse_zeph_summary_text(text: str) -> pd.DataFrame:
    """
    Parse the DataFrame repr from Zeph's summary cell.
    Block 1: index  env_id  status  n_turbines  ppo_mean  ppo_std  baseline_mean
    Block 2: index  baseline_std  gain_pct  simulator
    """
    row1_re = re.compile(
        r"^\s*(\d+)\s+(\S+_Floris)\s+(ok|error)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )
    row2_re = re.compile(
        r"^\s*(\d+)\s+([\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+\S+"
    )

    block1 = {}
    block2 = {}
    in_block2 = False

    for line in text.splitlines():
        if "baseline_std" in line:
            in_block2 = True
            continue
        if not in_block2:
            m = row1_re.match(line)
            if m:
                block1[int(m.group(1))] = {
                    "env_id":        m.group(2),
                    "status":        m.group(3),
                    "ppo_mean":      float(m.group(5)),
                    "ppo_std":       float(m.group(6)),
                    "baseline_mean": float(m.group(7)),
                }
        else:
            m = row2_re.match(line)
            if m:
                block2[int(m.group(1))] = {
                    "baseline_std": float(m.group(2)),
                    "gain_pct":     float(m.group(3)),
                }

    if not block1:
        return pd.DataFrame()

    rows = []
    for idx, b1 in block1.items():
        row = dict(b1)
        if idx in block2:
            row.update(block2[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def load_zeph_results() -> pd.DataFrame:
    print("Loading Zeph branch results...")
    frames = []
    for scenario, nb_path in [
        ("Scenario 1", "RL_PPO/RL_PPO_Scenario_1_full.ipynb"),
        ("Scenario 2", "RL_PPO/RL_PPO_Scenario_2_full.ipynb"),
    ]:
        try:
            nb_bytes = git_show("Zeph", nb_path)
        except RuntimeError as e:
            print(f"  WARNING: {e}")
            continue
        nb = json.loads(nb_bytes)

        summary_df = None
        for cell_idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            for out in cell.get("outputs", []):
                if out.get("output_type") != "execute_result":
                    continue
                text = "".join(out.get("data", {}).get("text/plain", []))
                if "env_id" in text and "ppo_mean" in text and "baseline_mean" in text:
                    summary_df = _parse_zeph_summary_text(text)
                    if not summary_df.empty:
                        print(f"  {scenario}: found summary at cell {cell_idx} "
                              f"({len(summary_df)} envs)")
                    break
            if summary_df is not None and not summary_df.empty:
                break

        if summary_df is None or summary_df.empty:
            print(f"  WARNING: could not find/parse Zeph {scenario} summary")
            continue

        df = summary_df[summary_df["status"] == "ok"].copy()
        df["mean_gain_kw"] = df["ppo_mean"] - df["baseline_mean"]
        df["std_gain_kw"]  = df["ppo_std"]
        df["scenario"]     = scenario
        df["branch"]       = "Zeph"
        df["method"]       = "PPO"
        df = df.rename(columns={"ppo_mean": "mean_reward", "ppo_std": "std_reward"})
        frames.append(df[["branch", "scenario", "env_id", "method",
                           "mean_gain_kw", "std_gain_kw", "gain_pct",
                           "mean_reward", "std_reward", "baseline_mean"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Combine and save CSV
# ---------------------------------------------------------------------------

def build_combined(kevin_df: pd.DataFrame, zeph_df: pd.DataFrame) -> pd.DataFrame:
    parts = [df for df in [kevin_df, zeph_df] if not df.empty]
    if not parts:
        raise RuntimeError("No data collected from any branch.")
    combined = pd.concat(parts, ignore_index=True)
    combined["env_id"] = combined["env_id"].str.strip()
    return combined.sort_values(["scenario", "env_id", "method"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def make_bar_chart(df: pd.DataFrame, out_path: str) -> None:
    scenarios  = ["Scenario 1", "Scenario 2"]
    methods    = ["GP", "TabPFN", "GraphPFN", "PPO"]
    envs       = REPRESENTATIVE_ENVS
    env_labels = [e.replace("_Floris", "").replace("_", " ") for e in envs]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Mean Power Gain vs Do-Nothing Baseline  |  representative FLORIS environments",
        fontsize=12, fontweight="bold",
    )

    bar_width = 0.18
    x = np.arange(len(envs))
    offsets = np.linspace(-(len(methods) - 1) / 2, (len(methods) - 1) / 2, len(methods)) * bar_width

    for ax, scenario in zip(axes, scenarios):
        sc_df = df[df["scenario"] == scenario]
        for offset, method in zip(offsets, methods):
            gains, errs = [], []
            for env in envs:
                row = sc_df[(sc_df["env_id"] == env) & (sc_df["method"] == method)]
                if row.empty:
                    gains.append(float("nan"))
                    errs.append(0.0)
                else:
                    gains.append(float(row["gain_pct"].iloc[0]))
                    bl  = float(row["baseline_mean"].iloc[0])
                    std = float(row["std_gain_kw"].iloc[0])
                    errs.append(std / bl * 100 if bl > 0 else 0.0)

            color = METHOD_COLORS.get(method, "#888888")
            ax.bar(x + offset, gains, bar_width,
                   yerr=errs, capsize=3,
                   label=method, color=color, alpha=0.85, error_kw={"linewidth": 1})

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(scenario, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(env_labels, fontsize=9, rotation=45, ha="right")
        ax.set_ylabel("Gain over Do-Nothing (%)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved bar chart -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    kevin_df = load_kevin_results()
    zeph_df  = load_zeph_results()

    combined = build_combined(kevin_df, zeph_df)

    csv_path = os.path.join(OUTPUT_DIR, "combined_results.csv")
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved CSV -> {csv_path}  ({len(combined)} rows)")
    print(combined.groupby(["branch", "scenario"])["env_id"].count().rename("n_rows").to_string())

    chart_path = os.path.join(OUTPUT_DIR, "performance_bar_chart.png")
    make_bar_chart(combined, chart_path)


if __name__ == "__main__":
    main()
