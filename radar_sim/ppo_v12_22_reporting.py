"""Cross-arm V12.22 reporting. This module reads results; it never evaluates cubes.

Existing V12.21 figure families are produced separately by the shared runner.
These additional plots expose failures, fixed greedy progress and stochastic
training separately. Missing jobs/metrics are reported as missing, never zero.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIMARY_ARMS = ("A", "B", "C", "D", "E", "F")
ARMS = PRIMARY_ARMS + ("D0", "E0", "F0")
LABELS = {
    "A": "A: original16 / original state / PPO",
    "B": "B: proposed21 / original state / PPO",
    "C": "C: proposed21 / CFAR state / PPO",
    "D": "D: CFAR PPO + historical margin",
    "E": "E: CFAR PPO + compatible-set",
    "F": "F: CFAR PPO + expected lowest-miss risk",
    "D0": "D0: original state + historical margin (optional interaction control)",
    "E0": "E0: original state + compatible set (optional interaction control)",
    "F0": "F0: original state + expected risk (optional interaction control)",
}
COLORS = dict(zip(ARMS, plt.get_cmap("tab10").colors[:len(ARMS)]))
ALIASES = {
    "update": ("checkpoint", "optimizer_update", "updates"),
    "episode": ("episode_index", "rollout_index", "rollout"),
    "low_miss": ("lowest_misses", "lowest_miss", "low_misses", "total_lowest_misses", "low_miss_total"),
    "six_miss": ("six_misses", "six_threshold_misses", "total_six_misses", "six_miss_total"),
    "unmatched_excess": ("false_excess", "unmatched_inflation", "unmatched_mean_excess", "outside_gate_excess"),
    "nmse": ("cube_nmse", "mean_nmse", "cube_nmse_mean"),
    "bpcs": ("accounted_bpcs", "mean_bpcs", "rate_bpcs"),
    "reward": ("total_reward", "terminal_reward", "mean_reward", "task_reward"),
    "split": ("population",),
    "cube_id": ("cube_key", "frame_id", "cube_name"),
    "episode_id": ("traffic_episode", "episode_name"),
}
METRICS = (
    ("low_miss", "Lowest-threshold misses [reported count]"),
    ("six_miss", "Six-threshold misses [reported count]"),
    ("unmatched_excess", "Mean unmatched excess / cube vs original"),
    ("nmse", "Cube NMSE"),
    ("bpcs", "Accounted bits / complex sample"),
    ("reward", "Unchanged V12.21 reward"),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    return result if isinstance(result, dict) else {}


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for target, sources in ALIASES.items():
        if target not in frame:
            for source in sources:
                if source in frame:
                    frame[target] = frame[source]
                    break
    for field in ["update", "episode", *[m[0] for m in METRICS]]:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame


def _read_table(path: Path, arm: str, seed: int) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = _canonical(pd.read_csv(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    # File location is the authoritative arm/seed identity, not CSV labels.
    if "arm" in frame and not frame["arm"].astype(str).eq(arm).all():
        raise ValueError(f"Arm mismatch in {path}")
    if "seed" in frame and not pd.to_numeric(frame["seed"], errors="coerce").eq(seed).all():
        raise ValueError(f"Seed mismatch in {path}")
    frame["arm"], frame["seed"] = arm, seed
    if "split" in frame:
        forbidden = frame["split"].astype(str).str.upper().str.contains("TEST", regex=False)
        if forbidden.any():
            raise ValueError(f"V12.22 report refuses TEST rows: {path}")
    return frame


def _join(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if not frame.empty]
    return pd.concat(valid, ignore_index=True, sort=False) if valid else pd.DataFrame()


def _validation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "split" not in frame:
        return frame.copy()
    keep = frame["split"].astype(str).str.lower().isin(("validation", "val", "dev", "development"))
    return frame.loc[keep].copy()


def _save(fig: plt.Figure, target: Path, figures: list[str]) -> None:
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(target.name))


def _plot_progress(frame: pd.DataFrame, path: Path, figures: list[str], *, training: bool) -> None:
    if frame.empty:
        return
    x = "episode" if training and "episode" in frame else "update"
    if x not in frame:
        return
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for ax, (metric, ylabel) in zip(axes.flat, METRICS):
        if metric not in frame:
            ax.text(.5, .5, f"{metric} was not exported", ha="center", va="center", transform=ax.transAxes)
        else:
            for (arm, seed), part in frame.groupby(["arm", "seed"], sort=True):
                part = part.sort_values(x, kind="stable")
                ax.plot(part[x], part[metric], color=COLORS.get(arm, "black"),
                        alpha=.35 if training else .85, linewidth=.65 if training else 1.4,
                        marker=None if training else "o", markersize=3,
                        linestyle="-" if int(seed) % 2 else "--", label=f"{arm}, seed {seed}")
            if metric in ("low_miss", "six_miss", "unmatched_excess"):
                ax.axhline(0, color="black", linewidth=.7, linestyle=":")
            if metric == "nmse":
                ax.axhline(.05, color="crimson", linewidth=.8, linestyle="--", label="provisional NMSE 0.05")
        ax.set(xlabel="Stochastic primary rollout (raw)" if training else "Optimizer update (fixed greedy cubes)", ylabel=ylabel)
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("V12.22 stochastic training — no smoothing; changing cubes" if training else
                 "V12.22 fixed greedy VALIDATION — identical checkpoints; no TEST evaluation")
    _save(fig, path, figures)


def _selected_rows(checkpoints: pd.DataFrame, statuses: dict[tuple[str, int], dict]) -> pd.DataFrame:
    frame = _validation(checkpoints)
    if frame.empty or "update" not in frame:
        return pd.DataFrame()
    output = []
    for key, part in frame.groupby(["arm", "seed"], sort=True):
        status = statuses.get((str(key[0]), int(key[1])), {})
        update = status.get("selected_update", status.get("selected_checkpoint"))
        known = update is not None
        if not known:
            # Descriptive final checkpoint only; not silently a selected checkpoint.
            update = part["update"].max()
        row = part.loc[part["update"].eq(float(update))]
        if len(row) != 1:
            continue
        entry = row.iloc[0].to_dict()
        entry["selection_recorded"] = known
        entry["selection_description"] = "validation-selected" if known else "latest available, selection missing"
        output.append(entry)
    return pd.DataFrame(output)


def _paired(selected: pd.DataFrame) -> pd.DataFrame:
    """Exact paired-seed deltas, never unpaired averages disguised as pairing."""
    if selected.empty:
        return pd.DataFrame()
    rows = []
    for reference in ("A", "B", "C", "D0", "E0", "F0"):
        baseline = selected.loc[selected.arm.eq(reference)]
        for arm in ARMS:
            if arm == reference:
                continue
            if reference.endswith("0") and arm != reference[0]:
                continue
            candidate = selected.loc[selected.arm.eq(arm)]
            for seed in sorted(set(baseline.seed) & set(candidate.seed)):
                lhs, rhs = candidate.loc[candidate.seed.eq(seed)].iloc[0], baseline.loc[baseline.seed.eq(seed)].iloc[0]
                record = {"arm": arm, "reference": reference, "seed": int(seed),
                          "candidate_update": lhs["update"], "reference_update": rhs["update"],
                          "both_selections_recorded": bool(lhs.selection_recorded and rhs.selection_recorded)}
                for metric, _ in METRICS:
                    if metric in selected:
                        record[f"delta_{metric}"] = lhs.get(metric, np.nan) - rhs.get(metric, np.nan)
                rows.append(record)
    return pd.DataFrame(rows)


def _fidelity(per_cube: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if per_cube.empty or "nmse" not in per_cube:
        return pd.DataFrame()
    frame = _validation(per_cube)
    rows = []
    for (arm, seed), part in frame.groupby(["arm", "seed"], sort=True):
        chosen = selected.loc[selected.arm.eq(arm) & selected.seed.eq(seed)] if not selected.empty else pd.DataFrame()
        if "update" in part and not chosen.empty:
            part = part.loc[part["update"].eq(chosen.iloc[0]["update"])]
        elif "update" in part:
            part = part.loc[part["update"].eq(part["update"].max())]
        nmse = pd.to_numeric(part.nmse, errors="coerce").dropna()
        if nmse.empty:
            continue
        rows.append({"arm": arm, "seed": int(seed), "n_cubes": len(nmse),
                     "nmse_mean": nmse.mean(), "nmse_median": nmse.median(),
                     "nmse_p95": nmse.quantile(.95), "nmse_max": nmse.max(),
                     "nmse_fraction_gt_0_01": nmse.gt(.01).mean(),
                     "nmse_fraction_gt_0_05": nmse.gt(.05).mean(),
                     "nmse_failures_gt_0_05": int(nmse.gt(.05).sum())})
    return pd.DataFrame(rows)


def _plot_pairing(selected: pd.DataFrame, path: Path, figures: list[str]) -> None:
    if selected.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    shown = (("low_miss", "Lowest-threshold misses"), ("six_miss", "Six-threshold misses"),
             ("unmatched_excess", "Unmatched excess vs original"), ("bpcs", "Accounted bpcs"))
    present = [arm for arm in ARMS if selected.arm.eq(arm).any()]
    for ax, (metric, title) in zip(axes.flat, shown):
        for arm in present:
            part = selected.loc[selected.arm.eq(arm)]
            if metric in part:
                x = np.full(len(part), present.index(arm), dtype=float)
                offsets = np.linspace(-.10, .10, len(part)) if len(part) > 1 else 0
                ax.scatter(x + offsets, part[metric], color=COLORS[arm], s=38, zorder=3)
                for i, (_, row) in enumerate(part.iterrows()):
                    ax.annotate(f"s{int(row.seed)} / u{int(row['update'])}",
                                (x[i] + (offsets[i] if isinstance(offsets, np.ndarray) else 0), row[metric]),
                                xytext=(3, 5), textcoords="offset points", fontsize=6)
        ax.set(xticks=range(len(present)), xticklabels=present, title=title, xlabel="Arm; points are individual seeds")
        ax.grid(axis="y", alpha=.2)
        ax.axhline(0, color="black", linewidth=.7, linestyle=":")
    fig.suptitle("V12.22 VALIDATION checkpoint outcomes — no hidden seed averaging")
    _save(fig, path, figures)


def _plot_auxiliary(frame: pd.DataFrame, path: Path, figures: list[str]) -> None:
    if frame.empty or "update" not in frame:
        return
    choices = (
        (("compatible_probability_after", "compatible_probability", "compatible_mass", "compatible_set_mass"), "Probability mass of best measured preservation set"),
        (("expected_lowest_misses", "expected_lowest_misses_after", "expected_low_risk", "expected_lowest_miss_risk", "expected_risk"), "Expected measured lowest-threshold misses"),
        (("weighted_auxiliary_gradient_norm", "weighted_aux_gradient_norm", "auxiliary_gradient_norm", "aux_gradient_norm"), "Auxiliary actor-gradient norm"),
        (("query_local_kl_mean", "query_local_kl", "local_kl", "approx_kl"), "Query-local policy KL"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, (alternatives, title) in zip(axes.flat, choices):
        metric = next((name for name in alternatives if name in frame), None)
        if metric is not None:
            for (arm, seed), part in frame.groupby(["arm", "seed"], sort=True):
                part = part.sort_values("update", kind="stable")
                ax.plot(part["update"], pd.to_numeric(part[metric], errors="coerce"), color=COLORS.get(arm),
                        alpha=.7, linewidth=.8, label=f"{arm}, s{seed}")
        else:
            ax.text(.5, .5, "Metric not exported", transform=ax.transAxes, ha="center")
        ax.set(xlabel="Optimizer update", ylabel=title)
        ax.grid(alpha=.2)
    if axes.flat[0].get_legend_handles_labels()[0]:
        axes.flat[0].legend(fontsize=7)
    fig.suptitle("V12.22 conditional/auxiliary learning — diagnostics, not deployment refinement")
    _save(fig, path, figures)


def _episode_outcomes(per_cube: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Retain episode variation; do not average away individual cube failures."""
    if per_cube.empty:
        return pd.DataFrame()
    frame = _validation(per_cube)
    episode = "episode_id" if "episode_id" in frame else "scene_name" if "scene_name" in frame else None
    if episode is None:
        return pd.DataFrame()
    parts = []
    for (arm, seed), part in frame.groupby(["arm", "seed"], sort=True):
        chosen = selected.loc[selected.arm.eq(arm) & selected.seed.eq(seed)] if not selected.empty else pd.DataFrame()
        if "update" in part and not chosen.empty:
            part = part.loc[part["update"].eq(chosen.iloc[0]["update"])]
        elif "update" in part:
            part = part.loc[part["update"].eq(part["update"].max())]
        for name, group in part.groupby(episode, sort=True, dropna=False):
            row = {"arm": arm, "seed": seed, "episode_or_scene": name, "group_field": episode,
                   "n_cubes": len(group)}
            for metric in ("low_miss", "six_miss"):
                if metric in group:
                    row[metric] = pd.to_numeric(group[metric], errors="coerce").sum(min_count=1)
            for metric in ("unmatched_excess", "nmse", "bpcs"):
                if metric in group:
                    row[metric] = pd.to_numeric(group[metric], errors="coerce").mean()
            if "nmse" in group:
                row["nmse_failures_gt_0_05"] = int(pd.to_numeric(group.nmse, errors="coerce").gt(.05).sum())
            parts.append(row)
    return pd.DataFrame(parts)


def _plot_episodes(frame: pd.DataFrame, path: Path, figures: list[str]) -> None:
    if frame.empty:
        return
    episodes = sorted(frame.episode_or_scene.astype(str).unique())
    shown = (("low_miss", "Lowest-threshold misses [sum]"), ("six_miss", "Six-threshold misses [sum]"),
             ("unmatched_excess", "Mean unmatched excess / cube"), ("nmse_failures_gt_0_05", "Cubes with NMSE > 0.05 [count]"))
    fig, axes = plt.subplots(2, 2, figsize=(max(11, len(episodes) * .8), 9), layout="constrained")
    for ax, (metric, ylabel) in zip(axes.flat, shown):
        if metric in frame:
            for (arm, seed), part in frame.groupby(["arm", "seed"], sort=True):
                x = [episodes.index(str(name)) for name in part.episode_or_scene]
                offset = (ARMS.index(arm) - 2.5) * .045
                ax.scatter(np.array(x) + offset, part[metric], color=COLORS.get(arm),
                           s=27, alpha=.7, label=f"{arm}, s{seed}")
        ax.set(xticks=range(len(episodes)), xticklabels=episodes, ylabel=ylabel, xlabel="Validation episode (individual seeds)")
        ax.tick_params(axis="x", labelrotation=50, labelsize=7)
        ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.suptitle("V12.22 episode variation — selected greedy VALIDATION, failures retained")
    _save(fig, path, figures)


def _costs(out: Path, statuses: dict[tuple[str, int], dict]) -> pd.DataFrame:
    rows = []
    for (arm, seed), status in sorted(statuses.items()):
        record = {"arm": arm, "seed": seed}
        for field in ("physical_evaluations", "detector_evaluations", "elapsed_seconds", "runtime_seconds", "selected_update", "completed"):
            if field in status:
                record[field] = status[field]
        rows.append(record)
    ledger = out / "tables" / "physical_evaluation_ledger.csv"
    if ledger.is_file():
        # The source ledger is left intact; the report cites its exact location.
        table = pd.read_csv(ledger)
        total = pd.to_numeric(table["physical_evaluations"], errors="raise").sum() if "physical_evaluations" in table else None
        seconds = pd.to_numeric(table["seconds"], errors="raise").sum() if "seconds" in table else None
        rows.append({"arm": "ALL (ledger total, not additive to job rows)", "seed": "",
                     "physical_evaluations": total, "detector_processing_seconds": seconds, "ledger": str(ledger)})
    return pd.DataFrame(rows)


def _decision(out: Path, selected: pd.DataFrame, paired: pd.DataFrame, fidelity: pd.DataFrame,
              missing: list[str], profile: str) -> str:
    lines = ["# V12.22 validation-only component decision", "",
             "This report is observational. No TEST cube was evaluated by this reporting module.",
             "A completed run is not evidence of zero misses or superiority. Two seeds and a limited number of validation episodes limit certainty.", "",
             f"Profile: **{profile}**. Tiny profiles establish engineering correctness only.", "",
             "## Attribution", "",
             "- A → B changes the action library and its corresponding output/state descriptor dimensions.",
             "- B → C isolates the four CFAR observations with matched initial decisions.",
             "- C → D/E/F isolates each auxiliary; D/E/F are not PPO alone.",
             "- Optional D0/E0/F0 controls retain the matched auxiliary but zero the four CFAR inputs. Parent minus control deltas isolate the state/auxiliary interaction.",
             "- All arms retain the original V12.21 terminal reward. Baselines must remain independent.", ""]
    if missing:
        lines += ["## Incomplete evidence", "", *[f"- {item}" for item in missing], ""]
    lines += ["## KEEP / MODIFY / REMOVE", "",
              "The automated rule is deliberately conservative: KEEP requires complete recorded validation selections, at least two paired seeds, non-worse lowest/six-threshold misses and unmatched excess in every pair, a strict improvement in at least one preservation/unmatched measure, and no added provisional NMSE failures. Otherwise the component is MODIFY / not established; REMOVE is not inferred from a single noisy run.", "",
              "Accounted bpcs and computation cost are reported separately. A component is not selected using RDO or TEST performance.", ""]
    complete = not missing and profile == "full"
    comparisons = [("B", "A", "Expanded menu"), ("C", "B", "CFAR observations"),
                   ("D", "C", "Historical margin"), ("E", "C", "Compatible set"), ("F", "C", "Expected lowest-miss risk")]
    if not selected.empty:
        comparisons += [(arm, arm + "0", f"CFAR/auxiliary interaction for {arm}") for arm in ("D", "E", "F")
                        if selected.arm.eq(arm + "0").any()]
    for arm, reference, label in comparisons:
        part = paired.loc[paired.arm.eq(arm) & paired.reference.eq(reference)] if not paired.empty else pd.DataFrame()
        columns = [f"delta_{metric}" for metric in ("low_miss", "six_miss", "unmatched_excess")]
        good = complete and len(part) >= 2 and all(col in part for col in columns)
        if good:
            values = part[columns].apply(pd.to_numeric, errors="coerce")
            good = bool(part.both_selections_recorded.all() and values.notna().all().all()
                        and values.le(0).all().all() and values.lt(0).any().any())
        if good:
            good = not fidelity.empty
            for seed in part.seed:
                a = fidelity.loc[fidelity.arm.eq(arm) & fidelity.seed.eq(seed)] if not fidelity.empty else pd.DataFrame()
                b = fidelity.loc[fidelity.arm.eq(reference) & fidelity.seed.eq(seed)] if not fidelity.empty else pd.DataFrame()
                good = good and len(a) == 1 and len(b) == 1
                if good:
                    good = bool(a.iloc[0].nmse_failures_gt_0_05 <= b.iloc[0].nmse_failures_gt_0_05)
        lines.append(f"- **{label} ({arm} vs {reference}): {'KEEP as a candidate; inspect per-episode results and cost' if good else 'MODIFY / benefit not established by the available paired evidence'}." )
    lines += ["", "## Mandatory failure disclosure", ""]
    if selected.empty or "low_miss" not in selected:
        lines.append("Lowest-threshold preservation cannot be established: selected validation counts are missing.")
    else:
        for _, row in selected.iterrows():
            value = row.get("low_miss", np.nan)
            claim = "zero reported lowest-threshold misses" if pd.notna(value) and value == 0 else f"lowest-threshold misses = {value}; zero-miss objective NOT satisfied"
            lines.append(f"- {row.arm}, seed {int(row.seed)}, update {int(row['update'])}: {claim} ({row.selection_description}).")
    lines += ["", "The x-axis compression-saving percentage in inherited rate sweeps must be computed as 100 × (1 − accounted compressed bits / original source bits). It is not a quantizer-bit label, rate cap, or percentage of objects removed.",
              "", "Unmatched peaks are outside the declared GT gates; they are not automatically verified physical false alarms.",
              "Reconstruction NMSE > 0.05 is a provisional engineering failure, not a validated safety threshold. All failed cubes remain in the tables.",
              "The original reward's zero rate weight means expanded low-rate actions do not themselves induce rate saving.", ""]
    return "\n".join(lines)


def build_comparison_report(out: str | Path) -> dict[str, Any]:
    """Create cross-arm tables/figures from already-saved non-TEST artifacts."""
    out = Path(out).expanduser().resolve()
    if not out.is_dir():
        raise FileNotFoundError(out)
    report_dir = out / "comparison"
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    inventory = {name: [] for name in ("training", "checkpoints", "conditional_actions", "validation_per_cube", "auxiliary_updates")}
    statuses: dict[tuple[str, int], dict] = {}
    missing: list[str] = []
    discovered = []
    for job in sorted((out / "jobs").glob("*_seed*")):
        if not job.is_dir():
            continue
        arm, _, raw_seed = job.name.partition("_seed")
        if arm not in ARMS or not raw_seed.isdigit():
            continue
        seed = int(raw_seed)
        discovered.append((arm, seed))
        status = _read_json(job / "job_status.json")
        statuses[(arm, seed)] = status
        if status.get("completed") is not True:
            missing.append(f"{job.name}: completed receipt absent or false")
        for name in inventory:
            path = job / "tables" / f"{name}.csv"
            if name == "validation_per_cube" and not path.is_file():
                path = job / "tables" / "per_cube.csv"
            frame = _read_table(path, arm, seed)
            inventory[name].append(frame)
            if name in ("training", "checkpoints", "validation_per_cube") and frame.empty:
                missing.append(f"{job.name}: {name}.csv absent/empty")
    manifest = _read_json(out / "manifest.json") or _read_json(out / "MANIFEST.json")
    profile = str(manifest.get("profile", "unknown"))
    expected_seeds = manifest.get("seeds", sorted({seed for _, seed in discovered}))
    expected_arms = list(manifest.get("arms", list(PRIMARY_ARMS)))
    reserve = _read_json(out / "INTERACTION_RESERVE.json")
    reserved = next((reserve.get(field) for field in ("parent_arm", "interaction_arm", "chosen_arm", "arm")
                     if reserve.get(field) in ("D", "E", "F")), None)
    if reserved is not None:
        expected_arms.append(reserved + "0")
    for arm in expected_arms:
        for seed in expected_seeds:
            if (str(arm), int(seed)) not in discovered:
                missing.append(f"{arm}_seed{seed}: job absent")
    frames = {name: _join(parts) for name, parts in inventory.items()}
    for name, frame in frames.items():
        if not frame.empty:
            frame.to_csv(tables_dir / f"all_{name}.csv", index=False)
    checkpoints = frames["checkpoints"]
    if not checkpoints.empty:
        validation = _validation(checkpoints)
        if validation.empty:
            missing.append("No fixed greedy VALIDATION checkpoint rows were exported.")
        for metric, _ in METRICS:
            if metric not in validation:
                missing.append(f"Fixed VALIDATION checkpoint metric absent: {metric}")
    selected = _selected_rows(checkpoints, statuses)
    paired = _paired(selected)
    fidelity = _fidelity(frames["validation_per_cube"], selected)
    episodes = _episode_outcomes(frames["validation_per_cube"], selected)
    costs = _costs(out, statuses)
    for name, frame in (("selected_validation", selected), ("paired_validation_deltas", paired),
                        ("per_cube_fidelity_summary", fidelity), ("episode_outcomes", episodes),
                        ("execution_cost", costs)):
        frame.to_csv(tables_dir / f"{name}.csv", index=False)
    figures: list[str] = []
    _plot_progress(frames["training"], figures_dir / "stochastic_training_raw_v12_22.png", figures, training=True)
    _plot_progress(_validation(checkpoints), figures_dir / "fixed_greedy_validation_progress_v12_22.png", figures, training=False)
    _plot_pairing(selected, figures_dir / "paired_seed_validation_outcomes_v12_22.png", figures)
    _plot_episodes(episodes, figures_dir / "per_episode_validation_failures_v12_22.png", figures)
    aux = frames["auxiliary_updates"] if not frames["auxiliary_updates"].empty else frames["conditional_actions"]
    _plot_auxiliary(aux, figures_dir / "conditional_auxiliary_learning_v12_22.png", figures)
    (report_dir / "DECISION.md").write_text(_decision(out, selected, paired, fidelity, missing, profile), encoding="utf-8")
    status = {"completed": True, "status": "DESCRIPTIVE_REPORT_ONLY", "test_evaluated": False,
              "profile": profile, "jobs_found": len(discovered), "missing_evidence": missing,
              "figures": figures, "directory": str(report_dir),
              "historical_figure_families": "Generated independently by the shared runner; this module does not replace them."}
    (report_dir / "REPORT_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def _optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = _canonical(pd.read_csv(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "split" in frame and frame["split"].astype(str).str.upper().str.contains("TEST", regex=False).any():
        raise ValueError(f"V12.22 learning plots refuse TEST rows: {path}")
    return frame


def _numeric_column(frame: pd.DataFrame, names: tuple[str, ...]) -> tuple[str | None, pd.Series | None]:
    name = next((key for key in names if key in frame), None)
    return (name, pd.to_numeric(frame[name], errors="coerce")) if name else (None, None)


def _learning_panel(ax, frame: pd.DataFrame, names: tuple[str, ...], ylabel: str,
                    *, update_only: bool = False, color: str = "tab:blue") -> None:
    name, values = _numeric_column(frame, names)
    if name is None:
        ax.text(.5, .5, "Metric not exported", transform=ax.transAxes, ha="center", va="center")
    else:
        x = frame["update"] if update_only and "update" in frame else np.arange(1, len(frame) + 1)
        ax.plot(x, values, color=color, linewidth=.8, marker="." if update_only else None,
                markersize=3, alpha=.85)
    ax.set(xlabel="Optimizer update" if update_only else "Primary rollout (raw; changing TRAIN cubes)", ylabel=ylabel)
    ax.grid(alpha=.2)


def _conditional_learning_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"update", "primary_row", "q_value_unscaled", "old_probability", "post_update_probability"}
    if frame.empty or required - set(frame):
        return pd.DataFrame()
    rows = []
    for (update, context), part in frame.groupby(["update", "primary_row"], sort=True):
        if "feasible" in part:
            feasible = part.feasible.astype(str).str.lower().isin(("true", "1"))
            part = part.loc[feasible]
        q = pd.to_numeric(part.q_value_unscaled, errors="coerce").to_numpy(float)
        old = pd.to_numeric(part.old_probability, errors="coerce").to_numpy(float)
        after = pd.to_numeric(part.post_update_probability, errors="coerce").to_numpy(float)
        if not len(q) or not np.isfinite(np.r_[q, old, after]).all():
            continue
        if (old < 0).any() or (after < 0).any() or not np.isclose(old.sum(), 1., atol=1e-5) or not np.isclose(after.sum(), 1., atol=1e-5):
            raise ValueError("Conditional credit probabilities are not normalized over feasible actions.")
        best = np.isclose(q, q.max(), rtol=1e-10, atol=1e-8)
        rows.append(dict(update=int(update), primary_row=int(context), feasible_actions=len(q),
                         branch_q_range=float(np.ptp(q)), branch_q_std=float(np.std(q)),
                         best_tied_actions=int(best.sum()), best_probability_before=float(old[best].sum()),
                         best_probability_after=float(after[best].sum()),
                         best_probability_change=float(after[best].sum() - old[best].sum()),
                         expected_branch_q_change=float(after @ q - old @ q),
                         greedy_after_in_measured_best_set=bool(best[int(np.argmax(after))])))
    return pd.DataFrame(rows)


def plot_selected_rate_reports(ex: Any, summary: pd.DataFrame, directory: str | Path,
                               *, arm: str, selected_update: int):
    """Reuse measured rate plots, but never label initialization as trained gain.

    Only presentation text is adapted. No numeric data or inherited global
    function is changed; the original publication saver receives the figure.
    """
    from types import SimpleNamespace
    from matplotlib.text import Text
    from radar_sim.ppo_v12_21_reporting import plot_percentage_rate_reports

    caption = f"PPO ({arm}, update {int(selected_update)}" + (", initialization)" if selected_update == 0 else ")")

    def save(fig, path):
        for text in fig.findobj(match=Text):
            label = text.get_text()
            if "Trained PPO" in label:
                text.set_text(label.replace("Trained PPO", caption))
        return ex.save_publication_figure(fig, path)

    return plot_percentage_rate_reports(SimpleNamespace(save_publication_figure=save), summary, directory)


def plot_job_learning(ex: Any, job_folder: str | Path, report_folder: str | Path) -> dict[str, Any]:
    """Preserve five learning-figure families with V12.22 names and honest units.

    Only saved scalar tables are read. No model, detector, radar cube or active
    pipeline is imported. ``report_folder`` is the caller's figures directory.
    The supplied publication saver controls output format/resolution exactly as
    in the inherited reports. Missing diagnostics remain explicitly missing.
    """
    job, destination = Path(job_folder), Path(report_folder)
    destination.mkdir(parents=True, exist_ok=True)
    training = _optional_csv(job / "tables" / "training.csv")
    checkpoints = _optional_csv(job / "tables" / "checkpoints.csv")
    conditional = _optional_csv(job / "tables" / "conditional_action_credit.csv")
    status = _read_json(job / "job_status.json")
    identity = job.name
    sort_columns = [column for column in ("update", "episode", "rollout_index") if column in training]
    if sort_columns:
        training = training.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    updates = training.drop_duplicates("update").sort_values("update") if "update" in training else training
    targets: list[str] = []

    def save(fig, name):
        target = destination / name
        try:
            ex.save_publication_figure(fig, target)
        finally:
            plt.close(fig)
        targets.append(str(target))

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for ax, (names, label) in zip(axes.flat, (
        (("low_miss",), "Lowest-threshold misses [count / cube, RD+RA]"),
        (("unmatched_excess",), "Unmatched excess vs original [count / cube]"),
        (("reward",), "Unchanged V12.21 terminal reward"),
        (("value_loss", "value_mse_after"), "Critic target MSE [optimizer update]"),
        (("approximate_kl", "approx_kl"), "Categorical policy KL [optimizer update]"),
        (("bpcs",), "Accounted bits / complex sample"),
    )):
        update_only = names[0] in ("value_loss", "approximate_kl")
        _learning_panel(ax, updates if update_only else training, names, label, update_only=update_only)
        if names[0] in ("low_miss", "unmatched_excess"):
            ax.axhline(0, color="black", linestyle=":", linewidth=.7)
    fig.suptitle(f"V12.22 {identity}: stochastic PPO training diagnostics — raw observations")
    save(fig, "ppo_training_diagnostics_v12_22.png")

    validation = _validation(checkpoints)
    if "update" in validation:
        validation = validation.sort_values("update", kind="stable")
    selected = status.get("selected_update", status.get("selected_checkpoint"))
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for ax, (names, label) in zip(axes.flat, (
        (("low_miss",), "Lowest-threshold misses [sum over fixed cubes]"),
        (("unmatched_excess",), "Mean unmatched excess / cube vs original"),
        (("reward",), "Mean unchanged terminal reward"),
        (("six_miss",), "Six-threshold misses [sum over fixed cubes]"),
        (("bpcs",), "Mean accounted bits / complex sample"),
        (("nmse",), "Mean cube NMSE; individual failures reported separately"),
    )):
        _learning_panel(ax, validation, names, label, update_only=True, color="tab:orange")
        if selected is not None:
            ax.axvline(float(selected), color="tab:green", linewidth=1., linestyle="--", label=f"selected update {selected}")
            ax.legend(fontsize=7)
        if names[0] in ("low_miss", "six_miss", "unmatched_excess"):
            ax.axhline(0, color="black", linestyle=":", linewidth=.7)
    fig.suptitle(f"V12.22 {identity}: fixed greedy VALIDATION progress — no TEST evaluation")
    save(fig, "fixed_validation_learning_v12_22.png")

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for ax, (names, label) in zip(axes.flat, (
        (("native_actor_loss", "actor_loss", "policy_loss"), "Native actor surrogate loss (without auxiliary)"),
        (("value_loss", "value_mse_after"), "Critic target MSE"),
        (("native_ppo_loss", "ppo_loss"), "Native reported PPO loss (without auxiliary)"),
        (("policy_entropy", "entropy"), "Reported policy entropy"),
        (("approximate_kl", "approx_kl"), "Categorical policy KL"),
        (("clip_fraction",), "Old-policy-weighted clipping fraction"),
    )):
        _learning_panel(ax, updates, names, label, update_only=True)
    fig.suptitle(f"V12.22 {identity}: PPO update health — losses/entropy need not be monotonic")
    save(fig, "ppo_losses_and_entropy_v12_22.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, (names, label) in zip(axes.flat, (
        (("rate_spectral_correlation",), "Corr(tile accounted rate, spectral concentration)"),
        (("rho_spectral_correlation",), "Corr(tile support fraction rho, spectral concentration)"),
        (("q_spectral_correlation",), "Corr(tile precision q, spectral concentration)"),
        (("unique_actions",), "Distinct actions used / reconstructed cube"),
    )):
        _learning_panel(ax, training, names, label)
        if names[0] != "unique_actions":
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color="black", linestyle=":", linewidth=.7)
    fig.suptitle(f"V12.22 {identity}: physical action adaptivity — no numeric action-ID correlation")
    save(fig, "ppo_physical_action_adaptivity_v12_22.png")

    summary = _conditional_learning_summary(conditional)
    summary.to_csv(destination / "conditional_learning_summary_v12_22.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, names, ylabel in (
        (axes[0, 0], ("branch_q_range",), "Measured branch Q max − min [reward units]"),
        (axes[0, 1], ("best_probability_before", "best_probability_after"), "Probability mass of all measured best-Q ties"),
        (axes[1, 0], ("best_probability_change",), "Post-update minus old best-set probability mass"),
        (axes[1, 1], ("expected_branch_q_change",), "Change in expected frozen branch Q [reward units]"),
    ):
        if summary.empty:
            ax.text(.5, .5, "Conditional credit not exported", transform=ax.transAxes, ha="center", va="center")
        else:
            for index, name in enumerate(names):
                ax.plot(summary["update"], summary[name], marker=".", markersize=3, linewidth=.8,
                        color=("tab:blue", "tab:orange")[index], label=name.replace("_", " "))
            if len(names) > 1:
                ax.legend(fontsize=7)
        ax.set(xlabel="Optimizer update (queried TRAIN context)", ylabel=ylabel)
        ax.grid(alpha=.2)
        if names[0] in ("best_probability_change", "expected_branch_q_change"):
            ax.axhline(0, color="black", linestyle=":", linewidth=.7)
    fig.suptitle(f"V12.22 {identity}: conditional credit audit — finite noisy Q estimates, not safe-action certificates")
    save(fig, "ppo_conditional_credit_audit_v12_22.png")
    return dict(figure_paths=targets, conditional_summary=str(destination / "conditional_learning_summary_v12_22.csv"),
                conditional_contexts=len(summary), selected_update=selected,
                test_evaluated=False, smoothing=False)
