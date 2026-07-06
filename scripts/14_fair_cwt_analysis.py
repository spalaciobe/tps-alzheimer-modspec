"""Análisis de la comparación JUSTA STFT vs CWT (cwt_fair) — fix del confound DSP.

Produce:
  - Métricas multi-seed de cwt_fair (media ± SD) para CNN y SVM (vanilla/gradcam).
  - DeLong AUC por seed + combinación Stouffer/Fisher, y Wilcoxon por seed, para:
        STFT vs cwt_fair  →  ¿aporta la resolución de portadora de la CWT a
                             IGUALDAD de eje de modulación?
        cwt  vs cwt_fair  →  cuantifica el CONFOUND DSP (cuánto del resultado
                             previo dependía del eje de modulación desalineado).

Funciona con seeds parciales (p.ej. solo seed 0 en el piloto). NO modifica nada:
solo lee fold_results.json de results/ y escribe results/fair_cwt_analysis.json.

Uso:
    python scripts/14_fair_cwt_analysis.py                 # seeds 0,1,2
    python scripts/14_fair_cwt_analysis.py --seeds 0       # piloto (1 seed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import combine_pvalues

from src.evaluate import compute_metrics
from src.stats import delong_test, wilcoxon_paired

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto/results")
OUT = BASE / "fair_cwt_analysis.json"


def cnn_metrics(method: str, seed: int) -> dict | None:
    p = BASE / f"{method}_200_seed{seed}" / "fold_results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if len(d) < 65:
        return None
    y_true = np.array([r["true"] for r in d])
    y_pred = np.array([r["pred"] for r in d])
    y_score = np.array([r["score"] for r in d])
    return {
        **compute_metrics(y_true, y_pred, y_score=y_score),
        "scores": y_score.tolist(),
        "subjects": [r["test_subject"] for r in d],
        "y_true": y_true.tolist(),
    }


def svm_metrics(method: str, seed: int, saliency: str) -> dict | None:
    sal_tag = "_vanilla" if saliency == "vanilla" else ""
    d_dir = BASE / f"svm_{method}_200_seed{seed}{sal_tag}_perfold"
    p = d_dir / "fold_results.json"
    s = d_dir / "summary.json"
    if not s.exists():
        return None
    summary = json.loads(s.read_text())
    if not p.exists():
        return {**summary}
    d = json.loads(p.read_text())
    y_true = np.array([r["true"] for r in d])
    y_score = np.array([r["score"] for r in d])
    return {
        **summary,
        "scores": y_score.tolist(),
        "subjects": [r["test_subject"] for r in d],
        "y_true": y_true.tolist(),
    }


def aggregate(runs, keys=("accuracy", "f1_macro", "sensitivity", "specificity", "auc")) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r is not None and k in r]
        if not vals:
            continue
        arr = np.array(vals)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": [float(v) for v in vals],
            "n_seeds": len(arr),
        }
    return out


def _paired_by_subject(a: dict, b: dict):
    """Alinea scores/labels de dos runs por sujeto común."""
    by_a = dict(zip(a["subjects"], a["scores"]))
    by_b = dict(zip(b["subjects"], b["scores"]))
    truth = dict(zip(a["subjects"], a["y_true"]))
    common = sorted(set(by_a) & set(by_b))
    sa = np.array([by_a[s] for s in common])
    sb = np.array([by_b[s] for s in common])
    y = np.array([truth[s] for s in common])
    return y, sa, sb, common


def delong_per_seed_combined(runs_a, runs_b) -> dict:
    """DeLong dentro de cada seed + combinación Stouffer/Fisher entre seeds."""
    per_seed, pvals = [], []
    for i, (a, b) in enumerate(zip(runs_a, runs_b)):
        if a is None or b is None or "scores" not in a or "scores" not in b:
            continue
        y, sa, sb, common = _paired_by_subject(a, b)
        if len(common) < 10:
            continue
        try:
            dl = delong_test(y, sa, sb)
        except Exception as e:
            per_seed.append({"seed": i, "error": str(e)})
            continue
        per_seed.append({
            "seed": i, "n_subjects": len(common),
            "auc_a": dl.get("auc_a"), "auc_b": dl.get("auc_b"),
            "auc_diff": dl.get("auc_diff"), "pvalue": dl.get("pvalue"),
        })
        if dl.get("pvalue") is not None:
            pvals.append(dl["pvalue"])
    out = {"per_seed": per_seed}
    if len(pvals) >= 2:
        st = combine_pvalues(pvals, method="stouffer")
        fi = combine_pvalues(pvals, method="fisher")
        out["combined"] = {
            "stouffer": {"statistic": float(st.statistic), "pvalue": float(st.pvalue)},
            "fisher": {"statistic": float(fi.statistic), "pvalue": float(fi.pvalue)},
            "n_seeds": len(pvals),
        }
    elif len(pvals) == 1:
        out["combined"] = {"single_seed_pvalue": float(pvals[0]), "n_seeds": 1}
    return out


def wilcoxon_per_seed(runs_a, runs_b) -> list:
    out = []
    for i, (a, b) in enumerate(zip(runs_a, runs_b)):
        if a is None or b is None or "scores" not in a or "scores" not in b:
            continue
        _y, sa, sb, _c = _paired_by_subject(a, b)
        try:
            out.append({"seed": i, **wilcoxon_paired(sa, sb)})
        except Exception:
            pass
    return out


def _fmt(agg, k):
    if k not in agg:
        return "  n/a"
    return f"{agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    seeds = args.seeds
    result = {"seeds": seeds, "cnn": {}, "svm": {}}

    print(f"=== Análisis comparación justa (cwt_fair) — seeds={seeds} ===\n")

    # ---- CNN ----
    runs = {m: [cnn_metrics(m, s) for s in seeds] for m in ("stft", "cwt", "cwt_fair")}
    for m in ("stft", "cwt", "cwt_fair"):
        result["cnn"][m] = {"agg": aggregate(runs[m])}
    result["cnn"]["stft_vs_cwt_fair"] = {
        "delong": delong_per_seed_combined(runs["stft"], runs["cwt_fair"]),
        "wilcoxon": wilcoxon_per_seed(runs["stft"], runs["cwt_fair"]),
    }
    result["cnn"]["cwt_vs_cwt_fair"] = {
        "delong": delong_per_seed_combined(runs["cwt"], runs["cwt_fair"]),
        "wilcoxon": wilcoxon_per_seed(runs["cwt"], runs["cwt_fair"]),
    }
    print("CNN end-to-end  AUC (media±SD):")
    for m in ("stft", "cwt", "cwt_fair"):
        print(f"   {m:9s}: {_fmt(result['cnn'][m]['agg'], 'auc')}")

    # ---- SVM (vanilla y gradcam) ----
    for sal in ("vanilla", "gradcam"):
        runs = {m: [svm_metrics(m, s, sal) for s in seeds] for m in ("stft", "cwt", "cwt_fair")}
        node = {m: {"agg": aggregate(runs[m])} for m in ("stft", "cwt", "cwt_fair")}
        node["stft_vs_cwt_fair"] = {
            "delong": delong_per_seed_combined(runs["stft"], runs["cwt_fair"]),
            "wilcoxon": wilcoxon_per_seed(runs["stft"], runs["cwt_fair"]),
        }
        node["cwt_vs_cwt_fair"] = {
            "delong": delong_per_seed_combined(runs["cwt"], runs["cwt_fair"]),
            "wilcoxon": wilcoxon_per_seed(runs["cwt"], runs["cwt_fair"]),
        }
        result["svm"][sal] = node
        print(f"\nSVM {sal}  Acc / AUC (media±SD):")
        for m in ("stft", "cwt", "cwt_fair"):
            print(f"   {m:9s}: acc {_fmt(node[m]['agg'], 'accuracy')}   auc {_fmt(node[m]['agg'], 'auc')}")
        dl = node["stft_vs_cwt_fair"]["delong"].get("combined", {})
        dlc = node["cwt_vs_cwt_fair"]["delong"].get("combined", {})
        print(f"   DeLong STFT vs cwt_fair : {json.dumps(dl)}")
        print(f"   DeLong cwt  vs cwt_fair : {json.dumps(dlc)}")

    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nGuardado: {OUT}")


if __name__ == "__main__":
    main()
