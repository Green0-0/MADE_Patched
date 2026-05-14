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
    load_experiment_metadata,
)
from made.evaluation.metrics import DiscoveryCurveMetrics

BASELINES_DIR = Path("results/baselines")

# Define the base names (prefixes) of the experiments you want to analyze
EXPERIMENT_PREFIXES = {
    "Random Generator (Baseline)": "random_generator_baseline_systems",
    "Chemeleon + MLIP": "chemeleon_mlip_ranking_chain_filter_systems",
    "Chemeleon + LLM Planner": "chemeleon_llm_planner_systems",
    "LLM Orchestrator": "llm_react_orchestrator_systems",
}

STRATEGY_ORDER = [
    "Random Generator (Baseline)",
    "Chemeleon + MLIP",
    "Chemeleon + LLM Planner",
    "LLM Orchestrator",
]

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


def dataset_key_from_metadata(metadata: dict | None, fallback_name: str) -> str:
    if metadata and isinstance(metadata, dict):
        systems_file = metadata.get("systems_file", "")
        if isinstance(systems_file, str):
            lowered = systems_file.lower()
            for key in ("ternary", "quaternary", "quinary"):
                if key in lowered:
                    return key
    lowered = fallback_name.lower()
    for key in ("ternary", "quaternary", "quinary"):
        if key in lowered:
            return key
    return "unknown"


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0}
    arr = np.array(values, dtype=float)
    n = len(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sem": float(np.std(arr) / np.sqrt(n)) if n > 0 else 0.0,
    }


def compute_af_ef_values(
    histories: list[list[dict]],
    baseline_histories: list[list[dict]],
    budget: int,
) -> tuple[list[float], list[float]]:
    if not histories or not baseline_histories:
        return [], []

    # Build averaged baseline curve for the dataset
    query_grid = np.arange(1, budget + 1, dtype=float)
    interpolated = []
    for history in baseline_histories:
        queries = [h.get("queries_used", 0) for h in history]
        discoveries = [h.get("num_newly_discovered_stable", 0) for h in history]
        if not queries:
            continue
        interp = np.interp(
            query_grid,
            np.array(queries, dtype=float),
            np.array(discoveries, dtype=float),
            left=0.0,
            right=float(discoveries[-1]),
        )
        interpolated.append(interp)

    if not interpolated:
        return [], []

    mean_curve = np.mean(np.stack(interpolated, axis=0), axis=0)
    fake_baseline = [
        {"queries_used": float(q), "num_newly_discovered_stable": float(d)}
        for q, d in zip(query_grid, mean_curve)
    ]

    avg_baseline_final = float(mean_curve[-1])
    target = int(avg_baseline_final)

    af_values: list[float] = []
    ef_values: list[float] = []

    for history in histories:
        ef = DiscoveryCurveMetrics.enhancement_factor(
            proposal_metrics_history=history,
            baseline_metrics_history=fake_baseline,
            metric_key="num_newly_discovered_stable",
        )
        ef_values.append(float(ef))

        if avg_baseline_final < 1.0:
            af_values.append(float(budget))
        else:
            af = DiscoveryCurveMetrics.acceleration_factor(
                proposal_metrics_history=history,
                baseline_metrics_history=fake_baseline,
                target_discoveries=target,
                metric_key="num_newly_discovered_stable",
            )
            af_values.append(float(af))

    return af_values, ef_values

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

    # 1. Load all results grouped by strategy and dataset
    strategy_data: dict[str, dict[str, dict[str, list]]] = {}
    for strategy_name, prefix in EXPERIMENT_PREFIXES.items():
        chunk_dirs = get_chunk_dirs(prefix)
        if not chunk_dirs:
            print(f"Skipping {strategy_name}: no directories found matching prefix '{prefix}'")
            continue

        per_dataset: dict[str, dict[str, list]] = {}
        for chunk_dir in chunk_dirs:
            metadata = load_experiment_metadata(chunk_dir)
            dataset_key = dataset_key_from_metadata(metadata, chunk_dir.name)

            hist, final = load_baseline_results(chunk_dir)
            dataset_entry = per_dataset.setdefault(
                dataset_key,
                {"histories": {}, "finals": {}},
            )
            for system_id, sys_hists in hist.items():
                dataset_entry["histories"].setdefault(system_id, []).extend(sys_hists)
            for system_id, sys_finals in final.items():
                dataset_entry["finals"].setdefault(system_id, []).extend(sys_finals)

        strategy_data[strategy_name] = per_dataset

    # 2. Compute common systems per dataset across all strategies
    datasets = set()
    for per_dataset in strategy_data.values():
        datasets.update(per_dataset.keys())

    common_systems: dict[str, set[str]] = {}
    for dataset_key in sorted(datasets):
        system_sets = []
        for strategy in STRATEGY_ORDER:
            systems = set(
                strategy_data.get(strategy, {})
                .get(dataset_key, {})
                .get("histories", {})
                .keys()
            )
            system_sets.append(systems)
        if system_sets:
            intersection = set.intersection(*system_sets)
            if intersection:
                common_systems[dataset_key] = intersection

    if not common_systems:
        print("No common systems found across strategies. Aborting.")
        return

    # 3. Process each strategy using dataset-matched pooled baselines
    baseline_name = "Random Generator (Baseline)"
    for strategy_name in STRATEGY_ORDER:
        if strategy_name not in strategy_data:
            continue

        print(f"Processing {strategy_name}...")

        combined_histories: list[list[dict]] = []
        combined_finals: list[dict] = []
        af_values: list[float] = []
        ef_values: list[float] = []

        for dataset_key, systems in common_systems.items():
            strategy_dataset = strategy_data.get(strategy_name, {}).get(dataset_key, {})
            baseline_dataset = strategy_data.get(baseline_name, {}).get(dataset_key, {})

            if not strategy_dataset or not baseline_dataset:
                continue

            dataset_histories: list[list[dict]] = []
            dataset_finals: list[dict] = []
            baseline_histories: list[list[dict]] = []

            for system_id in systems:
                dataset_histories.extend(
                    strategy_dataset.get("histories", {}).get(system_id, [])
                )
                dataset_finals.extend(
                    strategy_dataset.get("finals", {}).get(system_id, [])
                )
                baseline_histories.extend(
                    baseline_dataset.get("histories", {}).get(system_id, [])
                )

            if not dataset_histories:
                continue

            combined_histories.extend(dataset_histories)
            combined_finals.extend(dataset_finals)

            if strategy_name != baseline_name and baseline_histories:
                budget = 50
                af_vals, ef_vals = compute_af_ef_values(
                    dataset_histories, baseline_histories, budget
                )
                af_values.extend(af_vals)
                ef_values.extend(ef_vals)

        # Compute summary metrics from all aggregated final_metrics across datasets
        summary_metrics = aggregate_metrics(combined_finals)

        # Compute AUDC over the combined histories
        curve_metrics = compute_discovery_curve_metrics(
            combined_histories,
            baseline_histories=None,
            is_baseline=False,
        )
        
        # Extract metrics
        row = {"Policy": strategy_name}
        
        # Discovery Performance
        if strategy_name == baseline_name:
            row["AF"] = 1.0
            row["EF"] = 1.0
            af_sem = 0.0
            ef_sem = 0.0
        else:
            af_summary = summarize_values(af_values)
            ef_summary = summarize_values(ef_values)
            row["AF"] = af_summary.get("mean")
            row["EF"] = ef_summary.get("mean")
            af_sem = af_summary.get("sem", 0.0)
            ef_sem = ef_summary.get("sem", 0.0)
        
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
            row["AF"] = f"{row['AF']:.2f}({af_sem:.2f})"
        if row["EF"] is not None:
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
