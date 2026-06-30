"""
analyze.py — driver: metrics table + figures for the (currently synthetic)
shoulder-abduction dataset. Real-data-ready: point `load()` at real estimates
and every number/figure regenerates unchanged.

Outputs (to ./figures_out/):
  error_vs_azimuth.{png,svg}      — H1 + correction effect
  bland_altman_bonelength.{png,svg}
  data_efficiency.{png,svg}       — the headline: ML needs data, ours needs none
And prints a metrics table to stdout.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# JEI figure spec: graph text >= 12 pt, high-resolution raster export.
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})
SAVE_DPI = 300
SAVE_EXTS = ("png", "svg", "tiff", "jpg")

import metrics as M
import dataset as D

METHODS = ["baseline_nose", "2d_trunk", "bonelength", "3d_world"]
LABELS = {"baseline_nose": "baseline (nose-ref, app)",
          "2d_trunk": "2D trunk-ref",
          "bonelength": "bone-length correction (ours)",
          "3d_world": "3D reference (world landmarks)"}
# Colorblind-safe (Okabe-Ito) palette + distinct markers; avoids red/green pairing per JEI.
STYLE = {"baseline_nose": ("#999999", "o"),
         "2d_trunk":      ("#E69F00", "s"),
         "bonelength":    ("#0072B2", "^"),
         "3d_world":      ("#009E73", "D")}
OUT = "figures_out"
TAG = "synthetic"


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        em = M.error_metrics(df[m].values, df["true_angle"].values)
        ba = M.bland_altman(df[m].values, df["true_angle"].values)
        icc = M.icc21(np.c_[df["true_angle"].values, df[m].values])
        rows.append({"method": LABELS[m], "n": em["n"], "bias": em["bias"],
                     "MAE": em["mae"], "RMSE": em["rmse"],
                     "LoA_width": ba["loa_high"] - ba["loa_low"], "ICC": icc})
    return pd.DataFrame(rows).set_index("method").round(2)


def fig_error_vs_azimuth(df: pd.DataFrame):
    plt.figure(figsize=(7, 4.5))
    azs = sorted(df["azimuth"].unique())
    has_subj = "person" in df.columns
    for m in METHODS:
        means, errs = [], []
        for az in azs:
            d = df[df["azimuth"] == az]
            if has_subj:
                pm = np.array([np.nanmean(np.abs(g[m] - g["true_angle"]))
                               for _, g in d.groupby("person")])
                pm = pm[np.isfinite(pm)]
                means.append(pm.mean())
                errs.append(pm.std(ddof=1) / np.sqrt(len(pm)) if len(pm) > 1 else 0.0)
            else:
                means.append(np.nanmean(np.abs(d[m] - d["true_angle"])))
                errs.append(0.0)
        c, mk = STYLE[m]
        if has_subj:
            plt.errorbar(azs, means, yerr=errs, marker=mk, color=c, capsize=3, label=LABELS[m])
        else:
            plt.plot(azs, means, marker=mk, color=c, label=LABELS[m])
    plt.xlabel("camera azimuth (deg from frontal)")
    plt.ylabel("mean absolute error (deg)")
    plt.title(f"Shoulder-abduction error vs viewpoint ({TAG})")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    for ext in SAVE_EXTS:
        plt.savefig(f"{OUT}/error_vs_azimuth.{ext}", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()


def fig_bland_altman(df: pd.DataFrame, method="bonelength"):
    a, b = df[method].values, df["true_angle"].values
    m = np.isfinite(a) & np.isfinite(b)
    mean, diff = (a[m] + b[m]) / 2, a[m] - b[m]
    ba = M.bland_altman(a, b)
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(mean, diff, s=8, alpha=0.3)
    for y, ls, lab in [(ba["bias"], "-", f"bias {ba['bias']:.1f}"),
                       (ba["loa_high"], "--", f"+1.96SD {ba['loa_high']:.1f}"),
                       (ba["loa_low"], "--", f"-1.96SD {ba['loa_low']:.1f}")]:
        plt.axhline(y, ls=ls, color="k", lw=1)
        plt.text(mean.max(), y, " " + lab, va="center", fontsize=10)
    ref = "goniometer" if "person" in df.columns else "true angle"
    plt.xlabel(f"mean of method & {ref} (deg)")
    plt.ylabel(f"method − {ref} (deg)")
    plt.title(f"Bland–Altman vs {ref} ({TAG})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    for ext in SAVE_EXTS:
        plt.savefig(f"{OUT}/bland_altman_bonelength.{ext}", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()


def _knn_predict(Xtr, ytr, Xte, k=15):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_, Xte_ = (Xtr - mu) / sd, (Xte - mu) / sd
    out = np.empty(len(Xte_))
    for i, x in enumerate(Xte_):
        d = np.sum((Xtr_ - x) ** 2, axis=1)
        out[i] = ytr[np.argsort(d)[:k]].mean()
    return out


def data_efficiency(df: pd.DataFrame, seed=42):
    """Headline analysis: a learned (kNN) correction needs labeled training data;
    the bone-length correction needs none. Plot ML test-MAE vs training size,
    with bonelength + 3D as zero-training reference lines."""
    feat_cols = ["feat_armlen", "feat_trunklen", "feat_ratio", "2d_trunk", "baseline_nose"]
    d = df.dropna(subset=feat_cols + ["true_angle", "bonelength", "3d_world"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d))
    n_test = int(0.3 * len(d))
    te, tr = idx[:n_test], idx[n_test:]
    Xte, yte = d.loc[te, feat_cols].values, d.loc[te, "true_angle"].values

    sizes = [10, 20, 50, 100, 200, 500, len(tr)]
    sizes = [s for s in sizes if s <= len(tr)]
    ml_mae = []
    for s in sizes:
        sub = tr[:s]
        pred = _knn_predict(d.loc[sub, feat_cols].values, d.loc[sub, "true_angle"].values, Xte)
        ml_mae.append(np.mean(np.abs(pred - yte)))

    bone_mae = np.nanmean(np.abs(d.loc[te, "bonelength"].values - yte))
    d3_mae = np.nanmean(np.abs(d.loc[te, "3d_world"].values - yte))

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, ml_mae, marker="o", color="#E69F00", label="learned kNN correction (needs labeled data)")
    plt.axhline(bone_mae, color="#0072B2", ls="-", lw=2,
                label=f"bone-length correction (ours, 0 training): {bone_mae:.1f}deg")
    plt.axhline(d3_mae, color="#009E73", ls=":", lw=1.5, label=f"3D reference: {d3_mae:.1f}deg")
    plt.xscale("log")
    plt.xlabel("ML training-set size (labeled clips)")
    plt.ylabel("test MAE (deg)")
    plt.title("Data efficiency: learned vs training-free correction")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    for ext in SAVE_EXTS:
        plt.savefig(f"{OUT}/data_efficiency.{ext}", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    return sizes, ml_mae, bone_mae, d3_mae


def run(df, out_dir, label):
    global OUT, TAG
    OUT = out_dir
    TAG = f"real data, {df['person'].nunique()} subjects" if "person" in df.columns else "synthetic"
    os.makedirs(OUT, exist_ok=True)
    print(f"[{label}] {len(df)} clips "
          f"({df.true_angle.nunique()} angles x {df.azimuth.nunique()} azimuths)\n")

    tbl = metrics_table(df)
    print("=== Per-method accuracy/agreement vs goniometer truth (synthetic) ===")
    print(tbl.to_string())

    red = M.paired_mae_reduction(df["2d_trunk"].values, df["bonelength"].values,
                                 df["true_angle"].values)
    print(f"\nMAE reduction 2D-trunk -> bone-length: {red['mae_base']:.2f} -> "
          f"{red['mae_new']:.2f} deg (improvement {red['reduction']:.2f}, "
          f"95% CI [{red['ci_low']:.2f}, {red['ci_high']:.2f}])")

    fig_error_vs_azimuth(df)
    fig_bland_altman(df)
    sizes, ml_mae, bone_mae, d3 = data_efficiency(df)
    print(f"\nData efficiency: ML kNN test-MAE by training size:")
    for s, e in zip(sizes, ml_mae):
        print(f"    n={s:>5}: {e:5.2f} deg")
    print(f"    bone-length (0 training): {bone_mae:.2f} deg | 3D reference: {d3:.2f} deg")
    print(f"\nfigures saved to ./{OUT}/")


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        csv = sys.argv[2] if len(sys.argv) > 2 else "real_estimates.csv"
        outdir = sys.argv[3] if len(sys.argv) > 3 else "figures_out_real"
        df = pd.read_csv(csv)
        usable = df["baseline_nose"].notna().sum()
        df = df[df["baseline_nose"].notna()].copy()
        print(f"REAL data: {usable} usable clips from {csv}")
        run(df, outdir, "REAL")
    else:
        df = D.generate_synthetic(mode="persp", noise2d=0.004, noise3d=0.01, reps=20, seed=42)
        df.to_csv("synthetic_dataset.csv", index=False)
        run(df, "figures_out", "SYNTHETIC")


if __name__ == "__main__":
    main()
