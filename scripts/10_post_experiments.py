"""Experimentos post-análisis para enriquecer el informe TFM.

1. Correlación 2D entre saliency maps (STFT vs CWT, Grad-CAM vs vanilla).
2. Consistencia de patches por fold (Jaccard entre folds).
3. Análisis por banda canónica (proporción de píxeles activos en δ/θ/α/β/γ).
4. Confounders AD vs HC (edad, MMSE, género) con t-test/chi2.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto")
SAL = BASE / "data/derivatives/saliency"
OUT = BASE / "results/post_experiments"
OUT.mkdir(exist_ok=True)


# ---------- 1. Correlación 2D entre saliency maps -------------------------

def load_saliency_diff(method: str, seed: int, sal: str) -> np.ndarray | None:
    sal_tag = "_vanilla" if sal == "vanilla" else ""
    p = SAL / f"{method}_200_seed{seed}{sal_tag}" / "saliency_diff.npy"
    if not p.exists():
        return None
    return np.load(p)


def pearson_2d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1e-12:
        return 0.0
    return float((a * b).sum() / den)


def saliency_correlation_analysis() -> dict:
    print("=== 1. Correlación 2D saliency ===", flush=True)
    result = {}
    # STFT vs CWT (mismo seed, misma saliency)
    for sal in ("gradcam", "vanilla"):
        corrs = []
        for s in (0, 1, 2):
            a = load_saliency_diff("stft", s, sal)
            b = load_saliency_diff("cwt", s, sal)
            if a is None or b is None: continue
            corrs.append(pearson_2d(a, b))
        result.setdefault("stft_vs_cwt", {})[sal] = {
            "values": corrs,
            "mean": float(np.mean(corrs)) if corrs else None,
            "std": float(np.std(corrs, ddof=1)) if len(corrs) > 1 else 0.0,
        }
        print(f"  STFTvsCWT ({sal}): r={np.mean(corrs):.3f}+/-{np.std(corrs, ddof=1):.3f}")

    # Grad-CAM vs vanilla (mismo método, mismo seed)
    for m in ("stft", "cwt"):
        corrs = []
        for s in (0, 1, 2):
            a = load_saliency_diff(m, s, "gradcam")
            b = load_saliency_diff(m, s, "vanilla")
            if a is None or b is None: continue
            corrs.append(pearson_2d(a, b))
        result.setdefault("gradcam_vs_vanilla", {})[m] = {
            "values": corrs,
            "mean": float(np.mean(corrs)) if corrs else None,
            "std": float(np.std(corrs, ddof=1)) if len(corrs) > 1 else 0.0,
        }
        print(f"  Grad-CAMvsvanilla ({m}): r={np.mean(corrs):.3f}+/-{np.std(corrs, ddof=1):.3f}")
    return result


# ---------- 2. Consistencia de patches por fold (Jaccard) ----------------

def jaccard(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Jaccard 2D: |∩| / |∪|. Une todos los patches en una máscara binaria."""
    m1 = mask1.any(axis=0).astype(bool) if mask1.ndim == 3 else mask1.astype(bool)
    m2 = mask2.any(axis=0).astype(bool) if mask2.ndim == 3 else mask2.astype(bool)
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter / union) if union > 0 else 0.0


def patches_consistency() -> dict:
    print("\n=== 2. Consistencia de patches (Jaccard entre folds) ===", flush=True)
    result = {}
    for method in ("stft", "cwt"):
        for sal in ("gradcam", "vanilla"):
            for s in (0, 1, 2):
                sal_tag = "_vanilla" if sal == "vanilla" else ""
                pf_dir = SAL / f"{method}_200_seed{s}{sal_tag}" / "per_fold"
                if not pf_dir.exists(): continue
                masks = []
                for fold in range(65):
                    p = pf_dir / f"patch_masks_fold{fold:02d}.npy"
                    if p.exists():
                        masks.append(np.load(p))
                if len(masks) < 5:
                    continue
                # Sample Jaccard sobre pares de folds (sample 100 pares para eficiencia)
                rng = np.random.default_rng(0)
                pairs_idx = list(combinations(range(len(masks)), 2))
                if len(pairs_idx) > 200:
                    pairs_idx = [pairs_idx[i] for i in rng.choice(len(pairs_idx), 200, replace=False)]
                jaccs = [jaccard(masks[i], masks[j]) for i, j in pairs_idx]
                key = f"{method}_s{s}_{sal}"
                result[key] = {
                    "mean": float(np.mean(jaccs)),
                    "std": float(np.std(jaccs, ddof=1)),
                    "median": float(np.median(jaccs)),
                    "n_pairs": len(jaccs),
                    "n_folds": len(masks),
                }
                print(f"  {key}: J={np.mean(jaccs):.3f}+/-{np.std(jaccs, ddof=1):.3f} "
                      f"(n_pairs={len(jaccs)})")
    return result


# ---------- 3. Análisis por banda canónica -------------------------------

def canonical_band_analysis() -> dict:
    """Para cada saliency map agregado, qué fracción de píxeles activos cae
    en cada banda canónica del eje y (frecuencia portadora)."""
    print("\n=== 3. Bandas canónicas en saliency ===", flush=True)
    bands = {
        "delta (0.5-4 Hz)": (0.5, 4),
        "theta (4-8 Hz)": (4, 8),
        "alpha (8-13 Hz)": (8, 13),
        "beta (13-30 Hz)": (13, 30),
        "gamma (30-45 Hz)": (30, 45),
    }
    # El modspec es 45×45, eje y de 0.5 a 45 Hz
    carrier_freqs = np.linspace(0.5, 45, 45)
    result = {}
    for method in ("stft", "cwt"):
        for sal in ("gradcam", "vanilla"):
            for s in (0, 1, 2):
                arr = load_saliency_diff(method, s, sal)
                if arr is None: continue
                abs_arr = np.abs(arr)
                thr = np.percentile(abs_arr, 90)
                active = abs_arr >= thr
                band_props = {}
                for name, (lo, hi) in bands.items():
                    rows = np.where((carrier_freqs >= lo) & (carrier_freqs < hi))[0]
                    if len(rows) == 0: continue
                    prop = float(active[rows].sum() / active.sum()) if active.sum() > 0 else 0.0
                    band_props[name] = prop
                result[f"{method}_s{s}_{sal}"] = band_props
    # Print summary
    print("  Proporción de píxeles top-10% por banda (media entre seeds):")
    for method in ("stft", "cwt"):
        for sal in ("gradcam", "vanilla"):
            keys = [k for k in result if k.startswith(f"{method}_") and k.endswith(f"_{sal}")]
            if not keys: continue
            for band in ["delta (0.5-4 Hz)", "theta (4-8 Hz)", "alpha (8-13 Hz)", "beta (13-30 Hz)", "gamma (30-45 Hz)"]:
                vals = [result[k].get(band, 0) for k in keys]
                print(f"    {method} {sal} {band}: {np.mean(vals):.2%}")
    return result


# ---------- 4. Confounders AD vs HC -------------------------------------

def confounders_analysis() -> dict:
    print("\n=== 4. Confounders AD vs HC ===", flush=True)
    tsv = BASE / "data/raw/ds004504/participants.tsv"
    if not tsv.exists():
        print(f"  No encontrado: {tsv}")
        return {}
    df = pd.read_csv(tsv, sep="\t")
    df = df.rename(columns={"participant_id": "subject_id", "Group": "group",
                            "Age": "age", "MMSE": "mmse", "Gender": "gender"})
    df = df[df["group"].isin(("A", "C"))]
    print(f"  Total sujetos: {len(df)}")
    ad = df[df["group"] == "A"]
    hc = df[df["group"] == "C"]
    result = {"n_ad": len(ad), "n_hc": len(hc)}

    # Edad
    if "age" in df.columns:
        t, p = stats.ttest_ind(ad["age"].dropna(), hc["age"].dropna())
        result["age"] = {
            "ad": {"mean": float(ad["age"].mean()), "std": float(ad["age"].std())},
            "hc": {"mean": float(hc["age"].mean()), "std": float(hc["age"].std())},
            "ttest": {"t": float(t), "pvalue": float(p)},
        }
        print(f"  Edad: AD {ad['age'].mean():.1f}+/-{ad['age'].std():.1f} "
              f"vs HC {hc['age'].mean():.1f}+/-{hc['age'].std():.1f} (t={t:.2f}, p={p:.4f})")

    # MMSE
    if "mmse" in df.columns:
        t, p = stats.ttest_ind(ad["mmse"].dropna(), hc["mmse"].dropna())
        result["mmse"] = {
            "ad": {"mean": float(ad["mmse"].mean()), "std": float(ad["mmse"].std())},
            "hc": {"mean": float(hc["mmse"].mean()), "std": float(hc["mmse"].std())},
            "ttest": {"t": float(t), "pvalue": float(p)},
        }
        print(f"  MMSE: AD {ad['mmse'].mean():.1f}+/-{ad['mmse'].std():.1f} "
              f"vs HC {hc['mmse'].mean():.1f}+/-{hc['mmse'].std():.1f} (t={t:.2f}, p={p:.4f})")

    # Género (chi-square)
    if "gender" in df.columns:
        crosstab = pd.crosstab(df["group"], df["gender"])
        chi2, p, dof, exp = stats.chi2_contingency(crosstab)
        result["gender"] = {
            "crosstab": crosstab.to_dict(),
            "chi2": {"chi2": float(chi2), "pvalue": float(p), "dof": int(dof)},
        }
        print(f"  Género (chi2): chi2={chi2:.2f}, p={p:.4f}")
        print(f"    {crosstab.to_dict()}")
    return result


# ---------- Main ---------------------------------------------------------

def main():
    full = {}
    full["saliency_correlation"] = saliency_correlation_analysis()
    full["patches_consistency"] = patches_consistency()
    full["canonical_bands"] = canonical_band_analysis()
    full["confounders"] = confounders_analysis()
    (OUT / "post_experiments.json").write_text(json.dumps(full, indent=2))
    print(f"\nGuardado: {OUT / 'post_experiments.json'}")


if __name__ == "__main__":
    main()
