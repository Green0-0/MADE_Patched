from pathlib import Path
import pandas as pd

from results_analysis_utils import (
    load_baseline_results,
    load_baseline_overall_summary,
    compute_discovery_curve_metrics,
)

# Fill in the paths to your actual results directories from the 4 SLURM scripts
# Example: "results/20260510-120601-orb-random"
RESULTS_DIRS = {
    "Random Generator (Baseline)": Path("results/YOUR_RANDOM_RUN_DIR_HERE"),
    "Chemeleon + MLIP": Path("results/YOUR_CHEM_MLIP_RUN_DIR_HERE"),
    "Chemeleon + LLM Planner": Path("results/YOUR_CHEM_LLM_RUN_DIR_HERE"),
    "LLM Orchestrator": Path("results/YOUR_LLM_ORCH_RUN_DIR_HERE"),
}

def analyze_results():
    table_data = []
    
    # 1. Load baseline for relative metrics (AF/EF)
    baseline_dir = RESULTS_DIRS["Random Generator (Baseline)"]
    baseline_histories = []
    if baseline_dir.exists():
        # Load baseline histories for all systems
        all_metrics_histories, _ = load_baseline_results(baseline_dir)
        # Flatten across all systems/episodes
        baseline_histories = [hist for sys_hists in all_metrics_histories.values() for hist in sys_hists]
    else:
        print(f"Warning: Baseline directory not found at {baseline_dir}. Relative metrics (AF and EF) will not be computed.")
    
    # 2. Process each strategy
    for strategy_name, result_dir in RESULTS_DIRS.items():
        if not result_dir.exists():
            print(f"Skipping {strategy_name}: directory {result_dir} not found.")
            continue
            
        print(f"Processing {strategy_name}...")
        
        # Load the overall summary for this strategy
        overall_summary = load_baseline_overall_summary(result_dir)
        summary_metrics = overall_summary.get("summary", {}) if overall_summary else {}
        
        # We also need to compute AF and EF if not present in the summary
        # Let's compute them manually across all systems using compute_discovery_curve_metrics
        all_metrics_histories, _ = load_baseline_results(result_dir)
        strategy_histories = [hist for sys_hists in all_metrics_histories.values() for hist in sys_hists]
        
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
        audc = summary_metrics.get("final/area_under_discovery_curve_normalized", {})
        row["AUDC"] = f"{audc.get('mean', 0):.3f}({audc.get('sem', 0):.3f})" if "mean" in audc else None
        
        # mSUN (novelty_stable_unique_novel_fraction)
        msun = summary_metrics.get("final/novelty_stable_unique_novel_fraction", {})
        row["mSUN"] = f"{msun.get('mean', 0):.3f}({msun.get('sem', 0):.3f})" if "mean" in msun else None
        
        # Discovery Diversity
        mean_comp_l1 = summary_metrics.get("final/diversity_stable_composition_l1_distance_mean", {})
        row["Mean Comp. L1"] = f"{mean_comp_l1.get('mean', 0):.2f}({mean_comp_l1.get('sem', 0):.2f})" if "mean" in mean_comp_l1 else None
        
        unique_comps = summary_metrics.get("final/diversity_stable_composition_unique_composition_count", {})
        row["Unique Comps"] = f"{unique_comps.get('mean', 0):.1f}({unique_comps.get('sem', 0):.1f})" if "mean" in unique_comps else None
        
        unique_sgs = summary_metrics.get("final/diversity_stable_structure_unique_spacegroups_count", {})
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
