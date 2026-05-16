#!/usr/bin/env python3
"""Diagnose AF/EF computation across baseline experiments.

This script scans results/baselines for experiment outputs and performs
consistency checks plus multiple AF/EF computation variants to identify
mismatches caused by baseline selection or curve construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from results_analysis_utils import extract_metrics_history, load_episode_trajectory
from made.evaluation.metrics import DiscoveryCurveMetrics

DIAGNOSTICS_DIR = ROOT_DIR / "results" / "diagnostics"

PAPER_TABLE = {
    "random_generator_baseline": {
        "AF": 1.00,
        "EF": 1.00,
        "AUDC": 0.115,
        "mSUN": 0.115,
        "Mean Comp. L1": 0.98,
        "Unique Comps": 6.3,
        "Unique SGs": 1.00,
    },
    "random_generator_diversity_planner_comp": {
        "AF": 0.95,
        "EF": 1.11,
        "AUDC": 0.110,
        "mSUN": 0.120,
        "Mean Comp. L1": 0.97,
        "Unique Comps": 6.6,
        "Unique SGs": 1.00,
    },
    "random_generator_llm_planner": {
        "AF": 1.20,
        "EF": 1.45,
        "AUDC": 0.123,
        "mSUN": 0.124,
        "Mean Comp. L1": 0.48,
        "Unique Comps": 5.5,
        "Unique SGs": 1.00,
    },
    "chemeleon_generative_baseline": {
        "AF": 1.72,
        "EF": 1.70,
        "AUDC": 0.192,
        "mSUN": 0.192,
        "Mean Comp. L1": 0.89,
        "Unique Comps": 10.4,
        "Unique SGs": 2.74,
    },
    "chemeleon_diversity_planner_comp": {
        "AF": 2.10,
        "EF": 1.97,
        "AUDC": 0.192,
        "mSUN": 0.207,
        "Mean Comp. L1": 0.96,
        "Unique Comps": 10.8,
        "Unique SGs": 3.69,
    },
    "chemeleon_llm_planner": {
        "AF": 3.9,
        "EF": 3.33,
        "AUDC": 0.273,
        "mSUN": 0.264,
        "Mean Comp. L1": 0.70,
        "Unique Comps": 10.4,
        "Unique SGs": 6.8,
    },
    "chemeleon_mlip_ranking_chain_filter": {
        "AF": 6.4,
        "EF": 5.3,
        "AUDC": 0.42,
        "mSUN": 0.39,
        "Mean Comp. L1": 0.84,
        "Unique Comps": 19.2,
        "Unique SGs": 3.25,
    },
    "llm_react_orchestrator": {
        "AF": 5.4,
        "EF": 6.0,
        "AUDC": 0.401,
        "mSUN": 0.400,
        "Mean Comp. L1": 0.71,
        "Unique Comps": 10.6,
        "Unique SGs": 10.4,
    },
}

FINAL_METRIC_KEYS = {
    "AUDC": "final/discovery_curve_area_under_discovery_curve_normalized",
    "mSUN": "final/novelty_stable_unique_novel_fraction",
    "Mean Comp. L1": "final/diversity_stable_composition_l1_distance_mean",
    "Unique Comps": "final/diversity_stable_composition_unique_composition_count",
    "Unique SGs": "final/diversity_stable_structure_unique_spacegroups_count",
}

STRATEGY_ORDER = [
    "random_generator_baseline",
    "random_generator_diversity_planner_comp",
    "random_generator_llm_planner",
    "chemeleon_generative_baseline",
    "chemeleon_diversity_planner_comp",
    "chemeleon_llm_planner",
    "chemeleon_mlip_ranking_chain_filter",
    "llm_react_orchestrator",
]

METRIC_ORDER = [
    "AF",
    "EF",
    "AUDC",
    "mSUN",
    "Mean Comp. L1",
    "Unique Comps",
    "Unique SGs",
]

MSUN_KEY = "final/novelty_stable_unique_novel_fraction"
TOTAL_DISCOVERED_KEY = "final/num_newly_discovered_structures"


@dataclass(frozen=True)
class EpisodeInfo:
    file_path: Path
    episode_index: int | None
    history: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    final_queries_used: int | None
    final_new_stable: int | None


@dataclass(frozen=True)
class ExperimentInfo:
    path: Path
    agent_config: str
    systems_file: str
    dataset_key: str
    budget: int | None
    num_episodes: int | None
    stability_tolerance: float | None
    systems: dict[str, list[EpisodeInfo]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose AF/EF metrics for MADE baseline experiments."
    )
    parser.add_argument(
        "--baselines-dir",
        default=str(ROOT_DIR / "results" / "baselines"),
        help="Path to results/baselines directory",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on episodes per system to speed up analysis",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def gather_strategy_episodes(
    experiments: list[ExperimentInfo],
) -> dict[str, list[EpisodeInfo]]:
    by_strategy: dict[str, list[EpisodeInfo]] = defaultdict(list)
    for exp in experiments:
        for episodes in exp.systems.values():
            by_strategy[exp.agent_config].extend(episodes)
    return by_strategy


def collect_metric_values(
    episodes: list[EpisodeInfo],
    key: str,
) -> list[float]:
    values: list[float] = []
    for ep in episodes:
        if not ep.final_metrics:
            continue
        value = ep.final_metrics.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def dataset_key_from_systems_file(systems_file: str) -> str:
    lower = systems_file.lower()
    for key in ("ternary", "quaternary", "quinary"):
        if key in lower:
            return key
    return "unknown"


def parse_stability_tolerance(config_overrides: list[str]) -> float | None:
    for item in config_overrides:
        if "environment.stability_tolerance=" in item:
            try:
                return float(item.split("environment.stability_tolerance=")[-1])
            except ValueError:
                return None
    return None


def iter_experiment_dirs(baselines_dir: Path) -> Iterable[Path]:
    if not baselines_dir.exists():
        return []
    for ts_dir in sorted(baselines_dir.iterdir()):
        if not ts_dir.is_dir():
            continue
        for exp_dir in sorted(ts_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if (exp_dir / "experiment_metadata.json").exists():
                yield exp_dir


def read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def extract_episode_info(episode_file: Path) -> EpisodeInfo | None:
    traj = load_episode_trajectory(episode_file)
    if traj is None:
        return None

    history = extract_metrics_history(traj) or []
    metrics = traj.get("metrics", {}) if isinstance(traj, dict) else {}

    episode_index = None
    try:
        if episode_file.stem.startswith("episode_"):
            episode_index = int(episode_file.stem.split("episode_")[-1])
    except ValueError:
        episode_index = None

    final_queries = None
    final_stable = None
    if isinstance(metrics, dict):
        final_queries = metrics.get("final/queries_used")
        final_stable = metrics.get("final/num_newly_discovered_stable")

    return EpisodeInfo(
        file_path=episode_file,
        episode_index=episode_index,
        history=history,
        final_metrics=metrics if isinstance(metrics, dict) else {},
        final_queries_used=final_queries,
        final_new_stable=final_stable,
    )


def extract_curve(history: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not history:
        return np.array([0.0]), np.array([0.0])

    queries = np.array([h.get("queries_used", 0) for h in history], dtype=float)
    discoveries = np.array(
        [h.get("num_newly_discovered_stable", 0) for h in history], dtype=float
    )

    if len(queries) == 0:
        return np.array([0.0]), np.array([0.0])

    unique_mask = np.concatenate(([True], np.diff(queries) > 0))
    queries = queries[unique_mask]
    discoveries = discoveries[unique_mask]

    if queries[0] > 0:
        queries = np.concatenate([[0.0], queries])
        discoveries = np.concatenate([[0.0], discoveries])

    return queries, discoveries


def infer_budget(episodes: list[EpisodeInfo]) -> int:
    candidates: list[int] = []
    for ep in episodes:
        if ep.final_queries_used is not None:
            candidates.append(int(ep.final_queries_used))
        if ep.history:
            last_query = ep.history[-1].get("queries_used")
            if last_query is not None:
                candidates.append(int(last_query))
            candidates.append(len(ep.history))
    return max(candidates) if candidates else 0


def build_baseline_maps(
    experiments: list[ExperimentInfo],
) -> tuple[
    list[list[dict[str, Any]]],
    dict[str, list[list[dict[str, Any]]]],
    dict[tuple[str, str, int | None, float | None], list[list[dict[str, Any]]]],
]:
    baseline_all: list[list[dict[str, Any]]] = []
    baseline_by_dataset: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    baseline_by_system: dict[
        tuple[str, str, int | None, float | None], list[list[dict[str, Any]]]
    ] = defaultdict(list)

    for exp in experiments:
        if exp.agent_config != "random_generator_baseline":
            continue
        for system_id, episodes in exp.systems.items():
            histories = [ep.history for ep in episodes if ep.history]
            if not histories:
                continue
            effective_budget = exp.budget or infer_budget(episodes)
            key = (exp.dataset_key, system_id, effective_budget, exp.stability_tolerance)
            baseline_by_system[key].extend(histories)
            baseline_by_dataset[exp.dataset_key].extend(histories)
            baseline_all.extend(histories)

    return baseline_all, baseline_by_dataset, baseline_by_system


def collect_system_sets(
    experiments: list[ExperimentInfo],
) -> dict[str, dict[str, set[str]]]:
    system_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for exp in experiments:
        for system_id in exp.systems.keys():
            system_sets[exp.agent_config][exp.dataset_key].add(system_id)
    return system_sets


def common_systems_by_dataset(
    system_sets: dict[str, dict[str, set[str]]],
    strategies: list[str],
) -> dict[str, set[str]]:
    common: dict[str, set[str]] = {}
    datasets = set()
    for strategy in strategies:
        datasets.update(system_sets.get(strategy, {}).keys())

    for dataset in sorted(datasets):
        sets = []
        for strategy in strategies:
            sets.append(system_sets.get(strategy, {}).get(dataset, set()))
        if not sets:
            continue
        common_set = set.intersection(*sets) if sets else set()
        if common_set:
            common[dataset] = common_set
    return common


def write_common_systems(common: dict[str, set[str]]) -> None:
    rows: list[dict[str, Any]] = []
    for dataset, systems in common.items():
        for system_id in sorted(systems):
            rows.append({"dataset": dataset, "system": system_id})
    if not rows:
        return
    path = DIAGNOSTICS_DIR / "common_systems_by_dataset.csv"
    write_csv(path, rows, ["dataset", "system"])
    print(f"Saved: {path}")


def build_avg_history(
    histories: list[list[dict[str, Any]]],
    budget: int,
) -> list[dict[str, float]]:
    if not histories:
        return []

    query_grid = np.arange(1, budget + 1, dtype=float)
    interpolated = []
    for history in histories:
        q, d = extract_curve(history)
        interp = np.interp(query_grid, q, d, left=0.0, right=d[-1] if len(d) else 0.0)
        interpolated.append(interp)

    mean_curve = np.mean(np.stack(interpolated, axis=0), axis=0)
    return [
        {"queries_used": float(q), "num_newly_discovered_stable": float(d)}
        for q, d in zip(query_grid, mean_curve)
    ]


def build_avg_baseline_history(
    baseline_histories: list[list[dict[str, Any]]],
    budget: int,
) -> list[dict[str, float]]:
    return build_avg_history(baseline_histories, budget)


def compute_af_from_mean_curve(
    histories: list[list[dict[str, Any]]],
    baseline_histories: list[list[dict[str, Any]]],
    budget: int,
) -> float | None:
    proposal_avg = build_avg_history(histories, budget)
    baseline_avg = build_avg_history(baseline_histories, budget)
    if not proposal_avg or not baseline_avg:
        return None

    avg_baseline_final = baseline_avg[-1]["num_newly_discovered_stable"]
    if avg_baseline_final < 1.0:
        return float(budget)

    target = int(avg_baseline_final)
    af = DiscoveryCurveMetrics.acceleration_factor(
        proposal_metrics_history=proposal_avg,
        baseline_metrics_history=baseline_avg,
        target_discoveries=target,
        metric_key="num_newly_discovered_stable",
    )
    return float(af)


def compute_af_bins_from_mean_curve(
    histories: list[list[dict[str, Any]]],
    baseline_histories: list[list[dict[str, Any]]],
    budget: int,
    bins: np.ndarray,
) -> float | None:
    proposal_avg = build_avg_history(histories, budget)
    baseline_avg = build_avg_history(baseline_histories, budget)
    if not proposal_avg or not baseline_avg:
        return None

    af_bins = DiscoveryCurveMetrics.acceleration_factor(
        proposal_metrics_history=proposal_avg,
        baseline_metrics_history=baseline_avg,
        performance_bins=bins,
        percentage=True,
        metric_key="num_newly_discovered_stable",
    )
    values = np.array(af_bins, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.mean(values))


def compute_af_ef_values(
    histories: list[list[dict[str, Any]]],
    baseline_histories: list[list[dict[str, Any]]],
    budget: int,
) -> tuple[list[float], list[float]]:
    if not histories or not baseline_histories:
        return [], []

    fake_baseline = build_avg_baseline_history(baseline_histories, budget)
    if not fake_baseline:
        return [], []

    avg_baseline_final = fake_baseline[-1]["num_newly_discovered_stable"]
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


def compute_af_ef_for_strategy(
    exp_list: list[ExperimentInfo],
    baseline_all: list[list[dict[str, Any]]],
    baseline_by_dataset: dict[str, list[list[dict[str, Any]]]],
    baseline_by_system: dict[tuple[str, str, int | None, float | None], list[list[dict[str, Any]]]],
    mode: str,
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    af_values: list[float] = []
    ef_values: list[float] = []
    missing: list[dict[str, Any]] = []

    for exp in exp_list:
        for system_id, episodes in exp.systems.items():
            histories = [ep.history for ep in episodes if ep.history]
            if not histories:
                continue

            budget = exp.budget or infer_budget(episodes)
            if not budget:
                continue

            baseline_histories: list[list[dict[str, Any]]] = []
            if mode == "global":
                baseline_histories = baseline_all
            elif mode == "dataset":
                baseline_histories = baseline_by_dataset.get(exp.dataset_key, [])
            elif mode == "system":
                key = (exp.dataset_key, system_id, budget, exp.stability_tolerance)
                baseline_histories = baseline_by_system.get(key, [])

            if not baseline_histories:
                missing.append(
                    {
                        "strategy": exp.agent_config,
                        "dataset": exp.dataset_key,
                        "system": system_id,
                        "budget": budget,
                        "stability_tolerance": exp.stability_tolerance,
                    }
                )
                continue

            af_vals, ef_vals = compute_af_ef_values(
                histories, baseline_histories, budget
            )
            af_values.extend(af_vals)
            ef_values.extend(ef_vals)

    return af_values, ef_values, missing


def summarize(values: list[float]) -> tuple[float, float, float, int] | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    sem = float(np.std(arr) / math.sqrt(len(arr)))
    return mean, std, sem, len(arr)


def format_stat(label: str, stats: tuple[float, float, float, int] | None) -> str:
    if stats is None:
        return f"{label}: n/a"
    mean, std, sem, n = stats
    return f"{label}: mean={mean:.3f}, std={std:.3f}, sem={sem:.3f}, n={n}"


def mean_sem(stats: tuple[float, float, float, int] | None) -> tuple[float | None, float | None]:
    if stats is None:
        return None, None
    mean, _, sem, _ = stats
    return mean, sem


def median_value(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.array(values, dtype=float)))


def collect_experiments(baselines_dir: Path, max_episodes: int | None) -> list[ExperimentInfo]:
    experiments: list[ExperimentInfo] = []

    for exp_dir in iter_experiment_dirs(baselines_dir):
        metadata = read_json(exp_dir / "experiment_metadata.json")
        agent_config = metadata.get("agent_config", "unknown")
        systems_file = metadata.get("systems_file", "unknown")
        dataset_key = dataset_key_from_systems_file(systems_file)
        budget = metadata.get("budget")
        num_episodes = metadata.get("num_episodes")
        stability_tolerance = parse_stability_tolerance(metadata.get("config_overrides", []))

        systems_dir = exp_dir / "systems"
        systems: dict[str, list[EpisodeInfo]] = {}

        if systems_dir.exists():
            for system_dir in sorted(systems_dir.iterdir()):
                if not system_dir.is_dir():
                    continue
                trajectories_dir = system_dir / "trajectories"
                if not trajectories_dir.exists():
                    continue

                episode_files = sorted(trajectories_dir.glob("episode_*.json"))
                if max_episodes is not None:
                    episode_files = episode_files[:max_episodes]

                episodes: list[EpisodeInfo] = []
                for episode_file in episode_files:
                    info = extract_episode_info(episode_file)
                    if info is not None:
                        episodes.append(info)
                if episodes:
                    systems[system_dir.name] = episodes

        experiments.append(
            ExperimentInfo(
                path=exp_dir,
                agent_config=agent_config,
                systems_file=systems_file,
                dataset_key=dataset_key,
                budget=budget,
                num_episodes=num_episodes,
                stability_tolerance=stability_tolerance,
                systems=systems,
            )
        )

    return experiments


def pick_metric_key(history: list[dict[str, Any]]) -> str:
    if not history:
        return "num_newly_discovered_stable"
    for key in ["novelty_stable_unique_novel_count", "num_newly_discovered_stable"]:
        for entry in history:
            if key in entry:
                return key
    return "num_newly_discovered_stable"


def build_baseline_episode_map(
    experiments: list[ExperimentInfo],
) -> dict[tuple[str, str, int | None, float | None], list[EpisodeInfo]]:
    baseline_map: dict[tuple[str, str, int | None, float | None], list[EpisodeInfo]] = defaultdict(list)
    for exp in experiments:
        if exp.agent_config != "random_generator_baseline":
            continue
        for system_id, episodes in exp.systems.items():
            effective_budget = exp.budget or infer_budget(episodes)
            key = (exp.dataset_key, system_id, effective_budget, exp.stability_tolerance)
            baseline_map[key].extend(sorted(
                [ep for ep in episodes if ep.episode_index is not None],
                key=lambda ep: ep.episode_index,
            ))
    return baseline_map


def run_checks(experiments: list[ExperimentInfo]) -> list[dict[str, Any]]:
    print("\n=== Episode Consistency Checks ===")
    missing_hist = 0
    mismatch_final = 0
    mismatch_budget = 0
    mismatch_queries = 0
    missing_final_queries = 0
    total_eps = 0
    mismatch_records: list[dict[str, Any]] = []

    for exp in experiments:
        for system_id, episodes in exp.systems.items():
            for ep in episodes:
                total_eps += 1
                if not ep.history:
                    missing_hist += 1
                    continue

                hist_last = ep.history[-1].get("num_newly_discovered_stable")
                if ep.final_new_stable is not None and hist_last is not None:
                    if int(hist_last) != int(ep.final_new_stable):
                        mismatch_final += 1
                        mismatch_records.append(
                            {
                                "strategy": exp.agent_config,
                                "system": system_id,
                                "episode_file": str(ep.file_path),
                                "history_last_new_stable": hist_last,
                                "final_new_stable": ep.final_new_stable,
                            }
                        )

                if exp.budget is not None and ep.final_queries_used is not None:
                    if int(ep.final_queries_used) != int(exp.budget):
                        mismatch_budget += 1

                hist_last_query = ep.history[-1].get("queries_used")
                if ep.final_queries_used is None:
                    missing_final_queries += 1
                elif hist_last_query is not None:
                    if int(hist_last_query) != int(ep.final_queries_used):
                        mismatch_queries += 1

    print(f"Total episodes: {total_eps}")
    print(f"Episodes missing metrics_history: {missing_hist}")
    print(f"Episodes with final stable mismatch: {mismatch_final}")
    print(f"Episodes with budget mismatch: {mismatch_budget}")
    print(f"Episodes with final query mismatch: {mismatch_queries}")
    print(f"Episodes missing final/queries_used: {missing_final_queries}")

    return mismatch_records


def compute_strategy_summaries(experiments: list[ExperimentInfo]) -> None:
    print("\n=== AF/EF Diagnostic Summary ===")

    # Build baseline maps
    baseline_by_system: dict[tuple[str, str, int | None, float | None], list[list[dict[str, Any]]]] = defaultdict(list)
    baseline_by_dataset: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    baseline_all: list[list[dict[str, Any]]] = []

    for exp in experiments:
        if exp.agent_config != "random_generator_baseline":
            continue
        for system_id, episodes in exp.systems.items():
            effective_budget = exp.budget or infer_budget(episodes)
            histories = [ep.history for ep in episodes if ep.history]
            key = (exp.dataset_key, system_id, effective_budget, exp.stability_tolerance)
            baseline_by_system[key].extend(histories)
            baseline_by_dataset[exp.dataset_key].extend(histories)
            baseline_all.extend(histories)

    # Group experiments by strategy
    by_strategy: dict[str, list[ExperimentInfo]] = defaultdict(list)
    for exp in experiments:
        by_strategy[exp.agent_config].append(exp)

    for strategy, exp_list in sorted(by_strategy.items()):
        if strategy == "random_generator_baseline":
            continue

        af_global: list[float] = []
        ef_global: list[float] = []
        af_by_dataset: list[float] = []
        ef_by_dataset: list[float] = []
        af_by_system: list[float] = []
        ef_by_system: list[float] = []

        missing_baseline = 0
        total_systems = 0

        for exp in exp_list:
            for system_id, episodes in exp.systems.items():
                total_systems += 1
                histories = [ep.history for ep in episodes if ep.history]
                if not histories:
                    continue

                budget = exp.budget or infer_budget(episodes)
                if not budget:
                    continue

                # Global baseline (paper_analysis style)
                if baseline_all:
                    af_vals, ef_vals = compute_af_ef_values(histories, baseline_all, budget)
                    af_global.extend(af_vals)
                    ef_global.extend(ef_vals)

                # Dataset baseline
                dataset_baseline = baseline_by_dataset.get(exp.dataset_key, [])
                if dataset_baseline:
                    af_vals, ef_vals = compute_af_ef_values(histories, dataset_baseline, budget)
                    af_by_dataset.extend(af_vals)
                    ef_by_dataset.extend(ef_vals)

                # Per-system baseline
                key = (exp.dataset_key, system_id, budget, exp.stability_tolerance)
                system_baseline = baseline_by_system.get(key, [])
                if system_baseline:
                    af_vals, ef_vals = compute_af_ef_values(histories, system_baseline, budget)
                    af_by_system.extend(af_vals)
                    ef_by_system.extend(ef_vals)
                else:
                    missing_baseline += 1

        print(f"\nStrategy: {strategy}")
        print(f"Systems processed: {total_systems}")
        if missing_baseline:
            print(f"Missing per-system baseline matches: {missing_baseline}")

        print(format_stat("AF global", summarize(af_global)))
        print(format_stat("EF global", summarize(ef_global)))
        print(format_stat("AF per dataset", summarize(af_by_dataset)))
        print(format_stat("EF per dataset", summarize(ef_by_dataset)))
        print(format_stat("AF per system", summarize(af_by_system)))
        print(format_stat("EF per system", summarize(ef_by_system)))


def build_tables_and_deviations(experiments: list[ExperimentInfo]) -> None:
    baseline_all, baseline_by_dataset, baseline_by_system = build_baseline_maps(
        experiments
    )

    strategy_episodes = gather_strategy_episodes(experiments)
    base_metric_stats: dict[str, dict[str, tuple[float, float, float, int] | None]] = defaultdict(dict)

    for strategy, episodes in strategy_episodes.items():
        for metric_name, key in FINAL_METRIC_KEYS.items():
            values = collect_metric_values(episodes, key)
            base_metric_stats[strategy][metric_name] = summarize(values)

    modes = ["global", "dataset", "system"]

    for mode in modes:
        print(f"\n=== Table 1 Replica ({mode} baseline) ===")
        rows: list[dict[str, Any]] = []
        deviation_rows: list[dict[str, Any]] = []
        missing_rows: list[dict[str, Any]] = []

        for strategy in STRATEGY_ORDER:
            exp_list = [exp for exp in experiments if exp.agent_config == strategy]
            if not exp_list:
                continue

            if strategy == "random_generator_baseline":
                af_stats = (1.0, 0.0, 0.0, 0)
                ef_stats = (1.0, 0.0, 0.0, 0)
                missing = []
            else:
                af_vals, ef_vals, missing = compute_af_ef_for_strategy(
                    exp_list,
                    baseline_all,
                    baseline_by_dataset,
                    baseline_by_system,
                    mode,
                )
                af_stats = summarize(af_vals)
                ef_stats = summarize(ef_vals)

            missing_rows.extend(missing)

            row: dict[str, Any] = {"strategy": strategy}
            af_mean, af_sem = mean_sem(af_stats)
            ef_mean, ef_sem = mean_sem(ef_stats)

            row["AF_mean"] = af_mean
            row["AF_sem"] = af_sem
            row["EF_mean"] = ef_mean
            row["EF_sem"] = ef_sem

            for metric_name in ["AUDC", "mSUN", "Mean Comp. L1", "Unique Comps", "Unique SGs"]:
                stat = base_metric_stats.get(strategy, {}).get(metric_name)
                mean, sem = mean_sem(stat)
                row[f"{metric_name}_mean"] = mean
                row[f"{metric_name}_sem"] = sem

            rows.append(row)

            # Deviations vs paper
            if strategy in PAPER_TABLE:
                dev: dict[str, Any] = {"strategy": strategy, "baseline_mode": mode}
                paper = PAPER_TABLE[strategy]

                if af_mean is not None:
                    dev["AF_delta"] = af_mean - paper["AF"]
                if ef_mean is not None:
                    dev["EF_delta"] = ef_mean - paper["EF"]

                for metric_name in ["AUDC", "mSUN", "Mean Comp. L1", "Unique Comps", "Unique SGs"]:
                    mean = row.get(f"{metric_name}_mean")
                    if mean is not None:
                        dev[f"{metric_name}_delta"] = mean - paper[metric_name]

                deviation_rows.append(dev)

            # Console print (compact)
            display_af = "n/a" if af_mean is None else f"{af_mean:.3f}"
            display_ef = "n/a" if ef_mean is None else f"{ef_mean:.3f}"
            display_audc = row.get("AUDC_mean")
            display_audc = "n/a" if display_audc is None else f"{display_audc:.3f}"
            display_msun = row.get("mSUN_mean")
            display_msun = "n/a" if display_msun is None else f"{display_msun:.3f}"
            print(
                f"{strategy}: AF={display_af}, EF={display_ef}, AUDC={display_audc}, mSUN={display_msun}"
            )

        # Write tables
        table_path = DIAGNOSTICS_DIR / f"table1_replica_{mode}.csv"
        if rows:
            fieldnames = ["strategy"]
            for metric_name in ["AF", "EF", "AUDC", "mSUN", "Mean Comp. L1", "Unique Comps", "Unique SGs"]:
                fieldnames.extend([f"{metric_name}_mean", f"{metric_name}_sem"])
            write_csv(table_path, rows, fieldnames)
            print(f"Saved: {table_path}")

        dev_path = DIAGNOSTICS_DIR / f"paper_deviations_{mode}.csv"
        if deviation_rows:
            fieldnames = ["strategy", "baseline_mode"] + [
                f"{metric}_delta" for metric in METRIC_ORDER
            ]
            write_csv(dev_path, deviation_rows, fieldnames)
            print(f"Saved: {dev_path}")

        if missing_rows:
            missing_path = DIAGNOSTICS_DIR / f"missing_baselines_{mode}.csv"
            fieldnames = ["strategy", "dataset", "system", "budget", "stability_tolerance"]
            write_csv(missing_path, missing_rows, fieldnames)
            print(f"Saved: {missing_path}")

        # Print deviations
        if deviation_rows:
            print(f"\n--- Deviations vs paper ({mode}) ---")
            for dev in deviation_rows:
                deltas = []
                for metric in METRIC_ORDER:
                    key = f"{metric}_delta"
                    if key in dev:
                        deltas.append(f"{metric}={dev[key]:+.3f}")
                delta_str = ", ".join(deltas) if deltas else "n/a"
                print(f"{dev['strategy']}: {delta_str}")


def build_matched_tables_and_deviations(experiments: list[ExperimentInfo]) -> None:
    baseline_all, baseline_by_dataset, baseline_by_system = build_baseline_maps(
        experiments
    )
    baseline_episode_map = build_baseline_episode_map(experiments)
    system_sets = collect_system_sets(experiments)
    common = common_systems_by_dataset(system_sets, STRATEGY_ORDER)
    if not common:
        print("\nNo common systems across strategies found.")
        return

    write_common_systems(common)

    print("\n=== Table 1 Replica (dataset baseline, matched systems) ===")
    rows: list[dict[str, Any]] = []
    deviation_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    alt_rows: list[dict[str, Any]] = []
    bins = np.linspace(0.1, 1.0, 10, dtype=float)

    for strategy in STRATEGY_ORDER:
        exp_list = [exp for exp in experiments if exp.agent_config == strategy]
        if not exp_list:
            continue

        filtered_episodes: list[EpisodeInfo] = []
        af_values: list[float] = []
        ef_values: list[float] = []
        af_mean_curve_values: list[float] = []
        af_bin_mean_values: list[float] = []
        af_episode_matched_values: list[float] = []
        ef_episode_matched_values: list[float] = []
        af_episode_system_mean_values: list[float] = []
        ef_episode_system_mean_values: list[float] = []

        for exp in exp_list:
            dataset = exp.dataset_key
            allowed_systems = common.get(dataset, set())
            if not allowed_systems:
                continue

            for system_id, episodes in exp.systems.items():
                if system_id not in allowed_systems:
                    continue
                filtered_episodes.extend(episodes)

                if strategy == "random_generator_baseline":
                    continue

                budget = exp.budget or infer_budget(episodes)
                if not budget:
                    continue

                baseline_histories: list[list[dict[str, Any]]] = []
                for sys_id in allowed_systems:
                    key = (dataset, sys_id, budget, exp.stability_tolerance)
                    baseline_histories.extend(baseline_by_system.get(key, []))

                baseline_key = (dataset, system_id, budget, exp.stability_tolerance)
                baseline_histories_system = baseline_by_system.get(baseline_key, [])
                baseline_episodes_system = baseline_episode_map.get(baseline_key, [])
                baseline_avg_system = build_avg_history(baseline_histories_system, budget)
                baseline_avg_final = None
                if baseline_avg_system:
                    baseline_avg_final = float(
                        baseline_avg_system[-1]["num_newly_discovered_stable"]
                    )

                if not baseline_histories:
                    missing_rows.append(
                        {
                            "strategy": exp.agent_config,
                            "dataset": dataset,
                            "system": system_id,
                            "budget": budget,
                            "stability_tolerance": exp.stability_tolerance,
                        }
                    )
                    continue

                histories = [ep.history for ep in episodes if ep.history]
                if not histories:
                    continue

                af_vals, ef_vals = compute_af_ef_values(
                    histories, baseline_histories, budget
                )
                af_values.extend(af_vals)
                ef_values.extend(ef_vals)

                af_mean_curve = compute_af_from_mean_curve(
                    histories, baseline_histories_system, budget
                )
                if af_mean_curve is not None:
                    af_mean_curve_values.append(af_mean_curve)

                af_bin_mean = compute_af_bins_from_mean_curve(
                    histories, baseline_histories_system, budget, bins
                )
                if af_bin_mean is not None:
                    af_bin_mean_values.append(af_bin_mean)

                if baseline_avg_system and baseline_avg_final is not None:
                    if baseline_avg_final < 1.0:
                        target = None
                    else:
                        target = int(baseline_avg_final)
                    for ep in episodes:
                        if not ep.history:
                            continue
                        if target is None:
                            af_episode_system_mean_values.append(float(budget))
                        else:
                            af_val = DiscoveryCurveMetrics.acceleration_factor(
                                proposal_metrics_history=ep.history,
                                baseline_metrics_history=baseline_avg_system,
                                target_discoveries=target,
                                metric_key="num_newly_discovered_stable",
                            )
                            af_episode_system_mean_values.append(float(af_val))

                        ef_val = DiscoveryCurveMetrics.enhancement_factor(
                            proposal_metrics_history=ep.history,
                            baseline_metrics_history=baseline_avg_system,
                            metric_key="num_newly_discovered_stable",
                        )
                        ef_episode_system_mean_values.append(float(ef_val))

                if strategy != "random_generator_baseline" and baseline_episodes_system:
                    for ep in episodes:
                        if ep.episode_index is None:
                            continue
                        if ep.episode_index >= len(baseline_episodes_system):
                            continue
                        baseline_ep = baseline_episodes_system[ep.episode_index]
                        if not ep.history or not baseline_ep.history:
                            continue
                        metric_key = pick_metric_key(ep.history)
                        baseline_final = 0.0
                        for entry in baseline_ep.history:
                            if metric_key in entry:
                                baseline_final = float(entry[metric_key])
                        if baseline_final < 1.0:
                            af_episode_matched_values.append(float(budget))
                        else:
                            af_val = DiscoveryCurveMetrics.acceleration_factor(
                                proposal_metrics_history=ep.history,
                                baseline_metrics_history=baseline_ep.history,
                                target_discoveries=int(baseline_final),
                                metric_key=metric_key,
                            )
                            af_episode_matched_values.append(float(af_val))

                        ef_val = DiscoveryCurveMetrics.enhancement_factor(
                            proposal_metrics_history=ep.history,
                            baseline_metrics_history=baseline_ep.history,
                            metric_key=metric_key,
                        )
                        ef_episode_matched_values.append(float(ef_val))

        row: dict[str, Any] = {"strategy": strategy}

        if strategy == "random_generator_baseline":
            af_stats = (1.0, 0.0, 0.0, 0)
            ef_stats = (1.0, 0.0, 0.0, 0)
        else:
            af_stats = summarize(af_values)
            ef_stats = summarize(ef_values)

        af_mean, af_sem = mean_sem(af_stats)
        ef_mean, ef_sem = mean_sem(ef_stats)

        row["AF_mean"] = af_mean
        row["AF_sem"] = af_sem
        row["EF_mean"] = ef_mean
        row["EF_sem"] = ef_sem

        for metric_name, key in FINAL_METRIC_KEYS.items():
            values = collect_metric_values(filtered_episodes, key)
            stats = summarize(values)
            mean, sem = mean_sem(stats)
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_sem"] = sem

        rows.append(row)

        if strategy in PAPER_TABLE:
            dev: dict[str, Any] = {
                "strategy": strategy,
                "baseline_mode": "dataset_matched",
            }
            paper = PAPER_TABLE[strategy]

            if af_mean is not None:
                dev["AF_delta"] = af_mean - paper["AF"]
            if ef_mean is not None:
                dev["EF_delta"] = ef_mean - paper["EF"]

            for metric_name in ["AUDC", "mSUN", "Mean Comp. L1", "Unique Comps", "Unique SGs"]:
                mean = row.get(f"{metric_name}_mean")
                if mean is not None:
                    dev[f"{metric_name}_delta"] = mean - paper[metric_name]

            deviation_rows.append(dev)

        display_af = "n/a" if af_mean is None else f"{af_mean:.3f}"
        display_ef = "n/a" if ef_mean is None else f"{ef_mean:.3f}"
        display_audc = row.get("AUDC_mean")
        display_audc = "n/a" if display_audc is None else f"{display_audc:.3f}"
        display_msun = row.get("mSUN_mean")
        display_msun = "n/a" if display_msun is None else f"{display_msun:.3f}"
        print(
            f"{strategy}: AF={display_af}, EF={display_ef}, AUDC={display_audc}, mSUN={display_msun}"
        )

        alt_rows.append(
            {
                "strategy": strategy,
                "af_episode_mean": af_mean,
                "af_episode_median": median_value(af_values),
                "af_mean_curve_mean": float(np.mean(af_mean_curve_values))
                if af_mean_curve_values
                else None,
                "af_bin_mean": float(np.mean(af_bin_mean_values))
                if af_bin_mean_values
                else None,
                "af_episode_matched_mean": float(np.mean(af_episode_matched_values))
                if af_episode_matched_values
                else None,
                "ef_episode_matched_mean": float(np.mean(ef_episode_matched_values))
                if ef_episode_matched_values
                else None,
                "af_episode_system_mean": float(np.mean(af_episode_system_mean_values))
                if af_episode_system_mean_values
                else None,
                "ef_episode_system_mean": float(np.mean(ef_episode_system_mean_values))
                if ef_episode_system_mean_values
                else None,
            }
        )

    table_path = DIAGNOSTICS_DIR / "table1_replica_dataset_matched.csv"
    if rows:
        fieldnames = ["strategy"]
        for metric_name in ["AF", "EF", "AUDC", "mSUN", "Mean Comp. L1", "Unique Comps", "Unique SGs"]:
            fieldnames.extend([f"{metric_name}_mean", f"{metric_name}_sem"])
        write_csv(table_path, rows, fieldnames)
        print(f"Saved: {table_path}")

    dev_path = DIAGNOSTICS_DIR / "paper_deviations_dataset_matched.csv"
    if deviation_rows:
        fieldnames = ["strategy", "baseline_mode"] + [
            f"{metric}_delta" for metric in METRIC_ORDER
        ]
        write_csv(dev_path, deviation_rows, fieldnames)
        print(f"Saved: {dev_path}")

    if missing_rows:
        missing_path = DIAGNOSTICS_DIR / "missing_baselines_dataset_matched.csv"
        fieldnames = ["strategy", "dataset", "system", "budget", "stability_tolerance"]
        write_csv(missing_path, missing_rows, fieldnames)
        print(f"Saved: {missing_path}")

    if deviation_rows:
        print("\n--- Deviations vs paper (dataset_matched) ---")
        for dev in deviation_rows:
            deltas = []
            for metric in METRIC_ORDER:
                key = f"{metric}_delta"
                if key in dev:
                    deltas.append(f"{metric}={dev[key]:+.3f}")
            delta_str = ", ".join(deltas) if deltas else "n/a"
            print(f"{dev['strategy']}: {delta_str}")

    if alt_rows:
        alt_path = DIAGNOSTICS_DIR / "af_alternatives_dataset_matched.csv"
        fieldnames = [
            "strategy",
            "af_episode_mean",
            "af_episode_median",
            "af_mean_curve_mean",
            "af_bin_mean",
            "af_episode_matched_mean",
            "ef_episode_matched_mean",
            "af_episode_system_mean",
            "ef_episode_system_mean",
        ]
        write_csv(alt_path, alt_rows, fieldnames)
        print(f"Saved: {alt_path}")
        print("\n--- AF alternatives (dataset_matched, per-system baseline) ---")
        for row in alt_rows:
            parts = []
            for key in [
                "af_episode_mean",
                "af_episode_median",
                "af_mean_curve_mean",
                "af_bin_mean",
                "af_episode_matched_mean",
                "ef_episode_matched_mean",
                "af_episode_system_mean",
                "ef_episode_system_mean",
            ]:
                value = row.get(key)
                if value is not None:
                    parts.append(f"{key}={value:.3f}")
            parts_str = ", ".join(parts) if parts else "n/a"
            print(f"{row['strategy']}: {parts_str}")


def main() -> None:
    args = parse_args()
    baselines_dir = Path(args.baselines_dir)
    experiments = collect_experiments(baselines_dir, args.max_episodes)

    print("=== Experiment Inventory ===")
    print(f"Baselines directory: {baselines_dir}")
    print(f"Experiments found: {len(experiments)}")

    by_strategy: dict[str, int] = defaultdict(int)
    for exp in experiments:
        by_strategy[exp.agent_config] += 1
    for strategy, count in sorted(by_strategy.items()):
        print(f"{strategy}: {count}")

    mismatch_records = run_checks(experiments)
    if mismatch_records:
        mismatch_path = DIAGNOSTICS_DIR / "episode_mismatches.csv"
        write_csv(
            mismatch_path,
            mismatch_records,
            ["strategy", "system", "episode_file", "history_last_new_stable", "final_new_stable"],
        )
        print(f"Saved: {mismatch_path}")

    compute_strategy_summaries(experiments)
    build_tables_and_deviations(experiments)
    build_matched_tables_and_deviations(experiments)


if __name__ == "__main__":
    main()
