import numpy as np
from pathlib import Path
import pandas as pd
import sys
import os

# Ensure the notebooks directory is in the python path to import the utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "notebooks")))

from results_analysis_utils import (
    load_baseline_results,
    compute_discovery_curve_metrics,
)

BASELINES_DIR = Path("results/baselines")

# Define the base names (prefixes) of the experiments you want to analyze
EXPERIMENT_PREFIXES = {
    "Random Generator (Baseline)": "random_generator_baseline_systems",
    "Chemeleon + MLIP": "chemeleon_mlip_ranking_chain_filter_systems",
    "Chemeleon + LLM Planner": "chemeleon_llm_planner_systems",
    "LLM Orchestrator": "llm_react_orchestrator_systems",
}

def aggregate_metrics(per_episode: list[dict]) -> dict:
    """Replicates run_multi_systems.py metric aggregation across episodes."""
    def collect(name: str) -> list[float]:
        vals = []
        for m in per_episode:
            if name in m and isinstance(m[name], (int, float)):
                vals.append(float(m[name]))
        return vals

    all_keys = sorted({k for m in per_episode for k in m.keys()})
    summary = {}
    for k in all_keys:
        values = collect(k)
        if values:
            arr = np.array(values, dtype=float)
            n = len(arr)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "sem": float(np.std(arr) / np.sqrt(n)) if n > 0 else 0.0,
            }
    return summary

def get_chunk_dirs(prefix):
    """Finds all SLURM array job chunk directories for a given prefix."""
    chunk_dirs = []
    if not BASELINES_DIR.exists():
        return chunk_dirs
    for ts_dir in BASELINES_DIR.iterdir():
        if ts_dir.is_dir():
            for chunk_dir in ts_dir.iterdir():
                if chunk_dir.is_dir() and chunk_dir.name.startswith(prefix):
                    chunk_dirs.append(chunk_dir)
    return chunk_dirs

def analyze_results():
    table_data = []
    
    # 1. Load baseline for relative metrics (AF/EF)
    baseline_prefix = EXPERIMENT_PREFIXES["Random Generator (Baseline)"]
    baseline_chunk_dirs = get_chunk_dirs(baseline_prefix)
    
    baseline_histories = []
    if baseline_chunk_dirs:
        for chunk_dir in baseline_chunk_dirs:
            hist, _ = load_baseline_results(chunk_dir)
            baseline_histories.extend([h for sys_hists in hist.values() for h in sys_hists])
    else:
        print(f"Warning: Baseline chunks for '{baseline_prefix}' not found. Relative metrics (AF and EF) will not be computed.")
    
    # 2. Process each strategy
    for strategy_name, prefix in EXPERIMENT_PREFIXES.items():
        chunk_dirs = get_chunk_dirs(prefix)
        if not chunk_dirs:
            print(f"Skipping {strategy_name}: no directories found matching prefix '{prefix}'")
            continue
            
        print(f"Processing {strategy_name} ({len(chunk_dirs)} slurm tasks found)...")
        
        strategy_histories = []
        all_final_metrics_list = []
        
        for chunk_dir in chunk_dirs:
            hist, final = load_baseline_results(chunk_dir)
            strategy_histories.extend([h for sys_hists in hist.values() for h in sys_hists])
            all_final_metrics_list.extend([f for sys_finals in final.values() for f in sys_finals])
            
        # Compute summary metrics from all aggregated final_metrics across array chunks
        summary_metrics = aggregate_metrics(all_final_metrics_list)
        
        # Compute AF and EF
        curve_metrics = compute_discovery_curve_metrics(
            strategy_histories, 
            baseline_histories=baseline_histories if baseline_histories else None
        )
        
        # Extract metrics
        row = {"Policy": strategy_name}
        
        # Discovery Performance
        row["AF"] = curve_metrics.get("acceleration_factor", {}).get("mean", None)
        row["EF"] = curve_metrics.get("enhancement_factor", {}).get("mean", None)
        
        # Area Under Discovery Curve (normalized)
        audc = curve_metrics.get("area_under_discovery_curve_normalized", {})
        row["AUDC"] = f"{audc.get('mean', 0):.3f}({audc.get('sem', 0):.3f})" if "mean" in audc else None
        
        msun = summary_metrics.get("novelty_stable_unique_novel_fraction", {})
        row["mSUN"] = f"{msun.get('mean', 0):.3f}({msun.get('sem', 0):.3f})" if "mean" in msun else None
        
        # Discovery Diversity
        mean_comp_l1 = summary_metrics.get("diversity_stable_composition_l1_distance_mean", {})
        row["Mean Comp. L1"] = f"{mean_comp_l1.get('mean', 0):.2f}({mean_comp_l1.get('sem', 0):.2f})" if "mean" in mean_comp_l1 else None
        
        unique_comps = summary_metrics.get("diversity_stable_composition_unique_composition_count", {})
        row["Unique Comps"] = f"{unique_comps.get('mean', 0):.1f}({unique_comps.get('sem', 0):.1f})" if "mean" in unique_comps else None
        
        unique_sgs = summary_metrics.get("diversity_stable_structure_unique_spacegroups_count", {})
        row["Unique SGs"] = f"{unique_sgs.get('mean', 0):.2f}({unique_sgs.get('sem', 0):.2f})" if "mean" in unique_sgs else None
        
        # Format AF and EF if they exist
        if row["AF"] is not None:
            af_sem = curve_metrics.get("acceleration_factor", {}).get("sem", 0)
            row["AF"] = f"{row['AF']:.2f}({af_sem:.2f})"
        if row["EF"] is not None:
            ef_sem = curve_metrics.get("enhancement_factor", {}).get("sem", 0)
            row["EF"] = f"{row['EF']:.2f}({ef_sem:.2f})"
            
        table_data.append(row)

    # 3. Create DataFrame and Display
    if table_data:
        df = pd.DataFrame(table_data)
        print("\n" + "="*110)
        print("Replication of Table 1: Discovery Performance and Diversity")
        print("="*110)
        print(df.to_string(index=False))
        
        # Save to markdown and csv
        df.to_csv("paper_table1_replication.csv", index=False)
        df.to_markdown("paper_table1_replication.md", index=False)
        print(f"\nSaved results to paper_table1_replication.csv and .md")

if __name__ == "__main__":
    analyze_results()
