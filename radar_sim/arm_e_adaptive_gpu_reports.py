"""Read-only scientific reporting for the separately versioned adaptive Arm E.

No radar inference, tuning or training occurs here. Matplotlib/pandas run on the
CPU for file I/O and publication rendering; they are not the CUDA computation
path. Rates are analytically accounted bits, not a measured serialized stream.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_BPCS = 32.0
NOTE = "Analytically accounted bits; compression ratio = 32 / bpcs. No serialized-size claim."
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300,
}


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _save(fig, directory: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _require(frame: pd.DataFrame, names, context: str) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}")


def _correlation(path: Path, selected: pd.DataFrame, directory: Path) -> list[str]:
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        return []
    if frame.index.duplicated().any() or frame.columns.duplicated().any():
        raise ValueError(f"Duplicate feature names: {path}")
    if list(frame.index) != list(frame.columns):
        raise ValueError(f"Correlation matrix must be square with identical ordered names: {path}")
    values = frame.to_numpy(dtype=float)
    if not np.allclose(values, values.T, equal_nan=True):
        raise ValueError(f"Asymmetric correlation matrix: {path}")
    finite = values[np.isfinite(values)]
    if finite.size and np.max(np.abs(finite)) > 1 + 1e-8:
        raise ValueError(f"Correlation outside [-1,1]: {path}")
    drops = set()
    if not selected.empty:
        _require(selected, ("feature", "keep", "reason"), "feature_selection")
        kept = selected["keep"].astype(str).str.lower().isin(("true", "1", "yes"))
        drops = set(selected.loc[~kept, "feature"].astype(str))
    labels = [f"{name}{' [removed]' if name in drops else ''}" for name in frame.index]
    # Full legacy states can contain 62 variables. Keep every name readable in
    # the supplementary PDF rather than silently selecting a few variables.
    size = max(5.0, min(22.0, len(labels) * .23 + 3))
    fig, ax = plt.subplots(figsize=(size, size))
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#dddddd")
    im = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap=cmap)
    tick_size = 6 if len(labels) > 40 else 8 if len(labels) > 20 else 9
    ax.set_xticks(range(len(labels)), labels, rotation=90 if len(labels) > 25 else 65,
                  ha="right", fontsize=tick_size)
    ax.set_yticks(range(len(labels)), labels, fontsize=tick_size)
    kind = "Spearman rank" if "spearman" in path.stem else "Pearson linear"
    prefix = "Full legacy state" if path.stem.startswith("legacy_") else "Candidate adaptive state"
    ax.set_title(f"{prefix} — TRAIN-only {kind} correlation ({len(labels)} features)")
    if len(labels) <= 18:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(values[i,j]) > .65 else "black")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Correlation")
    fig.text(.01, .005, "Grey = undefined (e.g. constant feature). Correlation alone is not proof of irrelevance.", fontsize=7)
    return _save(fig, directory, path.stem)


def _training(frame: pd.DataFrame, directory: Path) -> list[str]:
    _require(frame, ("update", "record_name"), "training")
    metrics = (("low_miss", "Lowest-threshold misses / cube"),
               ("six_miss", "Six-threshold misses / cube"),
               ("unmatched_excess", "Unmatched excess / cube"),
               ("bpcs", "Accounted rate [bpcs]"),
               ("reward", "Training reward"), ("nmse", "Cube NMSE"))
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    grouped = frame.groupby("update", sort=True)
    for ax, (column, label) in zip(axes.flat, metrics):
        if column not in frame:
            ax.text(.5, .5, "Not exported", transform=ax.transAxes, ha="center")
        else:
            ax.scatter(frame["update"], frame[column], s=9, alpha=.25, label="Raw sampled cubes")
            means = grouped[column].mean()
            ax.plot(means.index, means, lw=1.3, color="#d95f02", label="Per-update mean (not smoothed)")
        ax.set_ylabel(label)
        ax.grid(alpha=.22)
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer update")
    axes[0,0].legend(loc="best")
    fig.suptitle("Stochastic TRAIN rollouts — changing cubes, not a fixed evaluation population")
    paths = _save(fig, directory, "stochastic_training")
    diagnostics = [(name, label) for name, label in (
        ("policy_loss", "PPO policy loss"), ("value_loss", "Value loss"),
        ("compatible_loss", "Compatible-action loss"), ("approx_kl", "Approximate KL"),
        ("entropy", "Policy entropy"), ("clip_fraction", "PPO clip fraction")) if name in frame]
    if diagnostics:
        fig, axes = plt.subplots(len(diagnostics), 1, figsize=(7, 1.65 * len(diagnostics)),
                                 sharex=True, constrained_layout=True, squeeze=False)
        for ax, (column, label) in zip(axes.flat, diagnostics):
            value = grouped[column].mean()
            ax.plot(value.index, value, marker=".", ms=3)
            ax.set_ylabel(label)
            ax.grid(alpha=.22)
        axes[-1,0].set_xlabel("Optimizer update")
        fig.suptitle("Optimizer diagnostics — a decreasing loss is not a preservation guarantee")
        paths += _save(fig, directory, "optimizer_diagnostics")
    return paths


def _validation(frame: pd.DataFrame, directory: Path) -> list[str]:
    _require(frame, ("update", "record_name"), "validation")
    if frame.duplicated(["update", "record_name"]).any():
        raise ValueError("validation: repeated cube/checkpoint rows")
    populations = [set(g.record_name) for _, g in frame.groupby("update")]
    if any(p != populations[0] for p in populations):
        raise ValueError("validation: checkpoint cube sets differ; fixed-population progress cannot be plotted")
    grouped = frame.groupby("update", sort=True)
    metrics = (("low_miss", "Total lowest-threshold misses", "sum"),
               ("six_miss", "Total six-threshold misses", "sum"),
               ("unmatched_excess", "Mean unmatched excess / cube", "mean"),
               ("bpcs", "Mean accounted rate [bpcs]", "mean"),
               ("reward", "Mean deterministic reward", "mean"),
               ("nmse", "Cube NMSE (mean and worst)", "mean"))
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    for ax, (column, label, reducer) in zip(axes.flat, metrics):
        if column not in frame:
            ax.text(.5, .5, "Not exported", transform=ax.transAxes, ha="center")
        else:
            values = getattr(grouped[column], reducer)(min_count=1) if reducer == "sum" else grouped[column].mean()
            ax.plot(values.index, values, "o-", ms=4, label=reducer)
            if column in ("low_miss", "six_miss"):
                ax.axhline(0, color="black", lw=.6, ls=":")
            if column == "nmse":
                worst = grouped[column].max()
                ax.plot(worst.index, worst, "s--", ms=3, label="maximum")
                ax.legend()
        ax.set_ylabel(label)
        ax.grid(alpha=.22)
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer update")
    fig.suptitle(f"Deterministic greedy validation — identical {len(populations[0])} cubes at each checkpoint")
    return _save(fig, directory, "fixed_validation_progress")


def aggregate_rate_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate real operating points; refuse duplicates or unmatched populations.

    Preservation pools original-supported object counts. AP/F1/recall/precision
    are equal-weight cube macros (undefined values remain missing). Six-threshold
    misses are summed, whereas unmatched counts are averaged over thresholds.
    """
    required = ("split", "method", "operating_point", "record_name", "bpcs", "nmse",
                "map_kind", "threshold", "original_supported", "preserved", "missed",
                "unmatched", "AP", "F1", "recall", "precision", "num_labels")
    _require(frame, required, "rate_sweep")
    frame = frame.copy()
    identity = ["split", "method", "operating_point", "record_name", "map_kind", "threshold"]
    if frame[identity].isna().any().any() or frame.duplicated(identity).any():
        raise ValueError("rate_sweep: missing identity or duplicate cube/operating-point/threshold")
    numeric = list(required[4:6]) + list(required[7:])
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame.bpcs).all() or (frame.bpcs <= 0).any():
        raise ValueError("rate_sweep: accounted rates must be finite and strictly positive")
    counts = ["original_supported", "preserved", "missed", "unmatched", "num_labels"]
    if not np.isfinite(frame[counts].to_numpy()).all() or (frame[counts] < 0).any().any():
        raise ValueError("rate_sweep: invalid detection counts")
    if not np.allclose(frame.preserved + frame.missed, frame.original_supported):
        raise ValueError("rate_sweep: preserved + missed must equal original_supported")
    for column in ("AP", "F1", "recall", "precision"):
        valid = frame[column].dropna()
        if ((valid < -1e-9) | (valid > 1 + 1e-9)).any():
            raise ValueError(f"rate_sweep: {column} outside [0,1]")
    # Same cube/threshold population is needed for every method and operating
    # point. This prevents connecting different scenes as a rate-response curve.
    for _, population in frame.groupby(["split", "map_kind"], sort=False):
        groups = [g for _, g in population.groupby(["method", "operating_point"], sort=False)]
        reference = set(zip(groups[0].record_name, groups[0].threshold))
        for group in groups[1:]:
            if set(zip(group.record_name, group.threshold)) != reference:
                raise ValueError("rate_sweep: methods/operating points have different cube/threshold populations")
        support = population.groupby(["record_name", "threshold"])["original_supported"].nunique()
        if (support > 1).any():
            raise ValueError("rate_sweep: original support changes between methods/operating points")
    ap_keys = ["split", "method", "operating_point", "record_name", "map_kind"]
    if (frame.groupby(ap_keys).AP.nunique(dropna=False) > 1).any():
        raise ValueError("rate_sweep: ranked-candidate AP must not vary with the CFAR reporting threshold")
    rows = []
    keys = ["split", "method", "operating_point", "map_kind"]
    for values, group in frame.groupby(keys, sort=True):
        base = dict(zip(keys, values))
        thresholds = sorted(group.threshold.unique())
        for view, subset in (("lowest", group[group.threshold == thresholds[0]]), ("threshold_mean", group)):
            per_cube = subset.groupby("record_name", sort=False)
            if (per_cube.bpcs.nunique() != 1).any() or (per_cube.nmse.nunique(dropna=False) != 1).any():
                raise ValueError("rate_sweep: cube rate/NMSE differs between thresholds")
            supported = float(subset.original_supported.sum())
            row = {**base, "view": view, "threshold_min": float(thresholds[0]),
                   "threshold_count": int(subset.threshold.nunique()), "n_cubes": len(per_cube),
                   "bpcs": per_cube.bpcs.first().mean(), "nmse": per_cube.nmse.first().mean(),
                   "nmse_max": per_cube.nmse.first().max(),
                   "original_supported": supported,
                   "preserved": float(subset.preserved.sum()), "missed": float(subset.missed.sum()),
                   "preservation": float(subset.preserved.sum()) / supported if supported > 0 else np.nan,
                   "unmatched": per_cube.unmatched.mean().mean()}
            for column in ("AP", "F1", "recall", "precision"):
                cube_values = per_cube[column].mean()
                row[column] = cube_values.mean()
                row[f"{column}_valid_cubes"] = int(cube_values.notna().sum())
            row["compression_ratio"] = RAW_BPCS / row["bpcs"]
            row["rate_saving_percent"] = 100 * (1 - row["bpcs"] / RAW_BPCS)
            rows.append(row)
    return pd.DataFrame(rows)


def _ratio_transform(values):
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(RAW_BPCS, values, out=np.full_like(values, np.inf), where=values != 0)


def _compression_ticks(left: float, right: float) -> np.ndarray:
    """Choose readable ratio labels by spacing in the *linear bitrate* axis.

    Powers of two are not equally spaced after the inverse transform. On a
    0--32 bpcs axis, 128x/64x/32x/16x otherwise pile up at the left edge.
    Prefer familiar 1x/2x/4x/8x anchors and a sparse high-compression label.
    """
    priority = (1, 2, 4, 8, 32, 16, 64, 128, 256, 512, 1024, 2048, 4096)
    gap = .075 * (right-left)
    selected, positions = [], []
    for ratio in priority:
        position = RAW_BPCS / ratio
        if left <= position <= right and all(abs(position-p) >= gap for p in positions):
            selected.append(ratio)
            positions.append(position)
    return np.asarray(sorted(selected), dtype=float)


def _rate_axis(ax) -> None:
    ax.set_xlabel("Accounted rate [bits / complex sample]")
    ax.grid(alpha=.22)
    left, right = ax.get_xlim()
    ax.set_xlim(max(.001, left), right)
    top = ax.secondary_xaxis("top", functions=(_ratio_transform, _ratio_transform))
    top.set_xlabel("Compression ratio [×], relative to 32 bpcs")
    lo, hi = ax.get_xlim()
    ticks = _compression_ticks(lo, hi)
    top.set_xticks(ticks)
    top.set_xticklabels([f"{x:g}×" for x in ticks])


def _safe_stem(*parts) -> str:
    return "_".join(str(p) for p in parts).replace("/", "_").replace("\\", "_").replace("=", "_")


def _rate_plots(summary: pd.DataFrame, directory: Path) -> list[str]:
    paths = []
    specs = (("preservation", "Original-supported label preservation [%]", True),
             ("AP", "Average precision (cube macro)", False),
             ("F1", "F1 score (cube macro)", False),
             ("recall", "Recall (cube macro)", False),
             ("precision", "Precision (cube macro)", False),
             ("unmatched", "Unmatched peaks / cube", False),
             ("missed", "Total original-supported misses", False),
             ("nmse", "Mean cube NMSE", False))
    for (split, kind), population in summary.groupby(["split", "map_kind"], sort=True):
        methods = list(population.method.unique())
        colors = dict(zip(methods, plt.get_cmap("tab10").colors * (1 + len(methods) // 10)))
        for metric, label, percent in specs:
            # AP is computed over ranked score candidates, not at an individual
            # threshold. Its repeated CSV value must not create two supposedly
            # different threshold-dependent AP panels.
            views = ("lowest",) if metric == "AP" else ("lowest", "threshold_mean")
            fig, axes = plt.subplots(1, len(views), figsize=(6.3 if metric == "AP" else 10, 3.7),
                                     constrained_layout=True, squeeze=False)
            for ax, view in zip(axes.flat, views):
                group = population[population.view == view]
                for method, points in group.groupby("method", sort=False):
                    points = points.sort_values("bpcs")
                    valid = points[np.isfinite(points[metric])]
                    if valid.empty:
                        continue
                    # One point is shown as one point, never an invented curve.
                    ax.plot(valid.bpcs, valid[metric] * (100 if percent else 1),
                            marker="o", ms=4, lw=1.5,
                            ls="-" if len(valid) > 1 else "none", color=colors[method], label=method)
                threshold = group.threshold_min.iloc[0]
                count = int(group.threshold_count.iloc[0])
                ax.set_title("All ranked detection candidates (not threshold-specific)" if metric == "AP" else
                    f"Lowest threshold = {threshold:g}" if view == "lowest" else f"Across {count} CFAR thresholds")
                ax.set_ylabel(label)
                _rate_axis(ax)
                if metric == "preservation":
                    ax.axhline(100, color="black", ls=":", lw=.7, label="Full original-supported preservation")
                    ax.set_ylim(top=101)
                elif metric in ("AP", "F1", "recall", "precision"):
                    ax.set_ylim(-.01, 1.01)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if handles:
                axes.flat[0].legend(handles, labels, loc="best")
            n = int(population.n_cubes.iloc[0])
            fig.suptitle(f"{str(split).upper()} • {kind} • {n} matched cubes • measured operating points")
            safe = _safe_stem(split, kind, metric)
            paths += _save(fig, directory, f"rate_{safe}")
    return paths


def pooled_ap_operating_points(population: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Attach accounted rate to independently computed exact population AP.

    Population AP must come from globally ranked detections. It cannot be
    reconstructed by averaging the per-cube AP values in rate_sweep.csv.
    """
    keys = ["split", "method", "operating_point", "map_kind"]
    _require(population, (*keys, "AP", "num_labels", "n_cubes"), "population_AP")
    if population[keys].isna().any().any() or population.duplicated(keys).any():
        raise ValueError("population_AP: missing identity or duplicate operating point")
    population = population.copy()
    for column in ("AP", "num_labels", "n_cubes"):
        population[column] = pd.to_numeric(population[column], errors="raise")
    valid = population.AP.dropna()
    if ((valid < 0) | (valid > 1) | ~np.isfinite(valid)).any():
        raise ValueError("population_AP: AP outside [0,1]")
    sizes = population[["num_labels", "n_cubes"]].to_numpy()
    if not np.isfinite(sizes).all() or (sizes < 0).any() or (sizes % 1 != 0).any():
        raise ValueError("population_AP: population sizes must be finite nonnegative integers")
    rates = summary[summary.view == "lowest"][keys + ["bpcs", "n_cubes", "compression_ratio"]]
    joined = population.merge(rates, how="outer", on=keys, validate="one_to_one",
                              suffixes=("", "_rate"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("population_AP: operating points do not match rate_sweep.csv")
    if not joined.n_cubes.eq(joined.n_cubes_rate).all():
        raise ValueError("population_AP: cube counts differ from rate_sweep.csv")
    for _, group in joined.groupby(["split", "map_kind"]):
        if group.num_labels.nunique(dropna=False) != 1:
            raise ValueError("population_AP: annotation count changes between operating points")
    return joined.drop(columns=["_merge", "n_cubes_rate"])


def _pooled_ap_plots(points: pd.DataFrame, directory: Path) -> list[str]:
    paths = []
    for (split, kind), group in points.groupby(["split", "map_kind"], sort=True):
        fig, ax = plt.subplots(figsize=(6.3, 4.1), constrained_layout=True)
        for method, values in group.groupby("method", sort=True):
            values = values[np.isfinite(values.AP)].sort_values("bpcs")
            if values.empty:
                continue
            ax.plot(values.bpcs, values.AP, "o", ms=4, lw=1.5,
                    ls="-" if len(values) > 1 else "none", label=method)
        _rate_axis(ax)
        ax.set_ylabel("Average precision (exact pooled ranking)")
        ax.set_ylim(-.01, 1.01)
        ax.set_title("All ranked detection candidates — not threshold-specific", pad=10)
        ax.legend(loc="best")
        fig.suptitle(f"{str(split).upper()} • {kind} • {int(group.n_cubes.iloc[0])} matched cubes")
        paths += _save(fig, directory, f"rate_{_safe_stem(split, kind)}_pooled_AP")
    return paths


def allocation_frequencies(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize measured final physical (rho,q) pairs, not policy commands."""
    _require(frame, ("split", "operating_point", "record_name", "tile", "rho", "q"), "allocations")
    frame = frame.copy()
    if frame.duplicated(["split", "operating_point", "record_name", "tile"]).any():
        raise ValueError("allocations: duplicate cube/tile/operating point")
    for column in ("rho", "q"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["rho", "q"]].to_numpy()).all():
        raise ValueError("allocations: non-finite action")
    if ((frame.rho < .01-1e-6) | (frame.rho > .5+1e-6)).any():
        raise ValueError("allocations: rho outside [0.01,0.5]")
    if ((frame.q < 1) | (frame.q > 12) | (frame.q % 1 != 0)).any():
        raise ValueError("allocations: q must be an integer in [1,12]")
    # Float32 arithmetic should not create several bins for the same physical
    # allocation. Four decimals are much finer than the requested 0.05 steps.
    frame["rho"] = frame.rho.round(4)
    frame["q"] = frame.q.astype(int)
    keys = ["split", "operating_point"]
    frequency = frame.groupby(keys + ["rho", "q"], sort=True).size().rename("count").reset_index()
    frequency["percent"] = frequency["count"] / frequency.groupby(keys)["count"].transform("sum") * 100
    coverage = []
    for values, group in frame.groupby(keys, sort=True):
        coverage.append(dict(zip(keys, values)) | dict(
            n_cubes=group.record_name.nunique(), n_allocations=len(group),
            observed_pairs=len(group[["rho", "q"]].drop_duplicates()),
            observed_rho_values=group.rho.nunique(), observed_q_values=group.q.nunique(),
            mean_rho=group.rho.mean(), mean_q=group.q.mean(),
            rho_lower_percent=group.rho.eq(.01).mean()*100,
            rho_upper_percent=group.rho.eq(.5).mean()*100,
            q_lower_percent=group.q.eq(1).mean()*100,
            q_upper_percent=group.q.eq(12).mean()*100))
    return frequency, pd.DataFrame(coverage)


def _allocation_plots(frequency: pd.DataFrame, coverage: pd.DataFrame, directory: Path) -> list[str]:
    paths = []
    for (split, point), group in frequency.groupby(["split", "operating_point"], sort=True):
        matrix = group.pivot(index="rho", columns="q", values="percent").fillna(0).reindex(columns=range(1,13), fill_value=0)
        cov = coverage[(coverage.split == split) & (coverage.operating_point == point)].iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(10, max(4.2, .23*len(matrix)+1.9)),
                                 constrained_layout=True, gridspec_kw={"width_ratios": [2.8, 1]})
        ax, marginal = axes
        image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
        ax.set_xticks(range(12), range(1,13))
        ax.set_yticks(range(len(matrix)), [f"{r:g}" for r in matrix.index])
        ax.set_xlabel("Quantization resolution q [bits / component], allowed 1–12")
        ax.set_ylabel("Observed retained-coefficient fraction ρ (allowed 0.01–0.50)")
        ax.set_title("Final physical allocations (not step-command IDs)")
        if len(matrix) <= 24:
            for r, row in enumerate(matrix.to_numpy()):
                for c, value in enumerate(row):
                    if value > .05:
                        ax.text(c, r, f"{value:.1f}", ha="center", va="center", fontsize=6,
                                color="white" if value > matrix.to_numpy().max()*.6 else "black")
        fig.colorbar(image, ax=ax, fraction=.04, pad=.02, label="Share of tile allocations [%]")
        fractions = matrix.sum(axis=1)
        marginal.barh(np.arange(len(matrix)), fractions.to_numpy(), color="#3274a1")
        marginal.set_yticks(np.arange(len(matrix)), [f"{r:g}" for r in matrix.index])
        marginal.invert_yaxis()
        marginal.set_xlabel("Tile allocations [%]")
        marginal.set_title("Retention marginal")
        marginal.grid(axis="x", alpha=.22)
        fig.suptitle(f"{str(split).upper()} • {point} • {int(cov.n_cubes)} cubes • "
                     f"{int(cov.observed_pairs)} observed (ρ,q) pairs")
        paths += _save(fig, directory, f"physical_actions_{_safe_stem(split, point)}")
    return paths


def command_frequencies(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize explicitly logged, count-weighted commands at each edit step."""
    keys = ["split", "operating_point", "step", "delta_rho", "delta_q"]
    _require(frame, (*keys, "count"), "step_commands")
    frame = frame.copy()
    for column in ("step", "delta_rho", "delta_q", "count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["step", "delta_rho", "delta_q", "count"]].to_numpy()).all():
        raise ValueError("step_commands: non-finite value")
    if (frame["count"] < 0).any() or (frame["count"] % 1 != 0).any():
        raise ValueError("step_commands: counts must be nonnegative integers")
    if (frame.step % 1 != 0).any() or (frame.step < 0).any():
        raise ValueError("step_commands: edit steps must be nonnegative integers")
    valid_rho = np.isclose(frame.delta_rho.to_numpy()[:,None], [-.05, 0, .05], atol=1e-6, rtol=0).any(axis=1)
    if not valid_rho.all() or not frame.delta_q.isin((-1,0,1)).all():
        raise ValueError("step_commands: expected delta_rho in {-0.05,0,+0.05}, delta_q in {-1,0,+1}")
    frame["delta_rho"] = frame.delta_rho.round(4)
    grouped = frame.groupby(keys, sort=True)["count"].sum().reset_index()
    totals = grouped.groupby(["split", "operating_point", "step"])["count"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("step_commands: each logged step must contain positive counts")
    grouped["percent_at_step"] = grouped["count"] / totals * 100
    return grouped


def _command_plots(frame: pd.DataFrame, directory: Path) -> list[str]:
    paths = []
    commands = [(r, q) for r in (-.05, 0, .05) for q in (-1,0,1)]
    for (split, point), group in frame.groupby(["split", "operating_point"], sort=True):
        matrix = group.pivot(index=["delta_rho", "delta_q"], columns="step", values="percent_at_step")
        matrix = matrix.reindex(pd.MultiIndex.from_tuples(commands, names=["delta_rho", "delta_q"])).fillna(0)
        fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        im = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=100)
        ax.set_yticks(range(9), [f"Δρ={r:+.2f}, Δq={q:+d}" for r,q in commands])
        # Production exports steps 1..ROUNDS; older engineering fixtures used
        # 0..ROUNDS-1. Only add one when a zero proves zero-based indexing.
        # Adding one unconditionally would mislabel production edits as 2..13.
        zero_based = bool((np.asarray(matrix.columns) == 0).any())
        labels = [str(int(s)+(1 if zero_based else 0)) for s in matrix.columns]
        ax.set_xticks(range(len(matrix.columns)), labels)
        ax.set_xlabel("Refinement-free policy edit step (one-based)")
        ax.set_ylabel("Commanded change (before boundary clipping)")
        ax.set_title("At a boundary, the realized change can be smaller than the command", fontsize=9)
        if len(matrix.columns) <= 20:
            for r, row in enumerate(matrix.to_numpy()):
                for c, value in enumerate(row):
                    if value > .05:
                        ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=6,
                                color="white" if value > 60 else "black")
        fig.colorbar(im, ax=ax, label="Tile commands at this step [%]")
        fig.suptitle(f"{str(split).upper()} • {point} • adaptive step-command frequencies")
        paths += _save(fig, directory, f"step_commands_{_safe_stem(split, point)}")
    return paths


def report(root: str | Path) -> dict:
    """Export a fresh report without touching input tables or existing figures."""
    root = Path(root).resolve()
    tables = root / "tables"
    if not tables.is_dir():
        raise FileNotFoundError(f"Missing input tables: {tables}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / "figures" / f"adaptive_gpu_report_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    files, missing = [], []
    selected = _read(tables / "feature_selection.csv")
    with plt.rc_context(STYLE):
        for name in ("feature_correlation_pearson", "feature_correlation_spearman"):
            path = tables / f"{name}.csv"
            if path.is_file():
                files += _correlation(path, selected, directory)
            else:
                missing.append(path.name)
        # These optional matrices audit all previous observation inputs without
        # pretending that the adaptive-state selector operated on their names.
        for name in ("legacy_feature_correlation_pearson", "legacy_feature_correlation_spearman"):
            path = tables / f"{name}.csv"
            if path.is_file():
                files += _correlation(path, pd.DataFrame(), directory)
        for name, plotter in (("training", _training), ("validation", _validation)):
            frame = _read(tables / f"{name}.csv")
            if frame.empty:
                missing.append(f"{name}.csv")
            else:
                files += plotter(frame, directory)
        sweep = _read(tables / "rate_sweep.csv")
        if not sweep.empty:
            summary = aggregate_rate_sweep(sweep)
            target = directory / "rate_operating_points.csv"
            summary.to_csv(target, index=False)
            files.append(str(target))
            files += _rate_plots(summary, directory)
            population_ap = _read(tables / "population_AP.csv")
            if not population_ap.empty:
                pooled = pooled_ap_operating_points(population_ap, summary)
                target = directory / "pooled_AP_operating_points.csv"
                pooled.to_csv(target, index=False)
                files.append(str(target))
                files += _pooled_ap_plots(pooled, directory)
        else:
            missing.append("rate_sweep.csv")
        allocations = _read(tables / "allocations.csv")
        if not allocations.empty:
            frequencies, coverage = allocation_frequencies(allocations)
            for name, frame in (("physical_action_frequencies", frequencies), ("physical_action_coverage", coverage)):
                target = directory / f"{name}.csv"
                frame.to_csv(target, index=False)
                files.append(str(target))
            files += _allocation_plots(frequencies, coverage, directory)
        commands = _read(tables / "step_commands.csv")
        if not commands.empty:
            frequencies = command_frequencies(commands)
            target = directory / "step_command_frequencies.csv"
            frequencies.to_csv(target, index=False)
            files.append(str(target))
            files += _command_plots(frequencies, directory)
    result = {"completed": True, "report_directory": str(directory), "files": files,
              "missing_inputs": missing, "analytical_rate_note": NOTE,
              "no_monotonic_improvement_assumed": True,
              "aggregation": "Preservation: pooled original-supported counts. AP/F1/recall/precision: cube macros; undefined excluded with valid-cube counts. Exact pooled AP: separately supplied globally ranked detections, never reconstructed from macro AP. AP is not threshold-specific. Unmatched: cube/threshold mean. Misses: total cube/threshold counts.",
              "action_coverage": "Observed physical (rho,q) pairs only; this is not a fraction of all feasible policies. Commands are measured separately and can be clipped at rho/q bounds.",
              "test_not_used_for_selection": "Reporting only; this module performs no model or feature selection."}
    path = directory / "REPORT_STATUS.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.out), indent=2))
