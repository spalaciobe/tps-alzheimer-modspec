"""Análisis multi-seed: media ± SD, intervalo CI95 bootstrap, tests pareados.

Para cada configuración (CNN/SVM × STFT/CWT × Grad-CAM/vanilla):
- Carga las 3 semillas (s0, s1, s2).
- Reporta media ± SD de acc, F1, sens, spec, AUC.
- Wilcoxon STFT vs CWT por seed y agregado.
- DeLong sobre pooled scores.
- Bootstrap CI95.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluate import compute_metrics
from src.stats import bootstrap_ci, delong_test, wilcoxon_paired

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto/results")
OUT = BASE / "multiseed_analysis.json"
SEEDS = [0, 1, 2]


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
        "y_pred": y_pred.tolist(),
    }


def svm_metrics(method: str, seed: int, saliency: str) -> dict | None:
    sal_tag = "_vanilla" if saliency == "vanilla" else ""
    p = BASE / f"svm_{method}_200_seed{seed}{sal_tag}_perfold" / "fold_results.json"
    s = BASE / f"svm_{method}_200_seed{seed}{sal_tag}_perfold" / "summary.json"
    if not s.exists():
        return None
    summary = json.loads(s.read_text())
    if not p.exists():
        return {**summary}
    d = json.loads(p.read_text())
    y_true = np.array([r["true"] for r in d])
    y_pred = np.array([r["pred"] for r in d])
    y_score = np.array([r["score"] for r in d])
    return {
        **summary,
        "scores": y_score.tolist(),
        "subjects": [r["test_subject"] for r in d],
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def aggregate(runs: list[dict | None], keys=("accuracy", "f1_macro", "sensitivity", "specificity", "auc")) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r is not None and k in r]
        if not vals:
            continue
        arr = np.array(vals)
        out[k] = {
            "values": [float(v) for v in vals],
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": len(arr),
        }
    return out


def wilcoxon_per_seed(stft_runs: list[dict], cwt_runs: list[dict], key: str = "scores") -> dict:
    """Wilcoxon STFT vs CWT por seed individual."""
    out = []
    for i, (a, b) in enumerate(zip(stft_runs, cwt_runs)):
        if a is None or b is None:
            continue
        sa = np.array(a[key])
        sb = np.array(b[key])
        n = min(len(sa), len(sb))
        if n == 0:
            continue
        try:
            w = wilcoxon_paired(sa[:n], sb[:n])
            out.append({"seed": i, **w})
        except Exception:
            pass
    return {"per_seed": out}


def delong_pooled(stft_runs: list[dict], cwt_runs: list[dict]) -> dict:
    """DeLong test sobre scores agrupados por sujeto (mediana entre seeds).

    NOTA: este DeLong "pooled" opera sobre la mediana entre seeds, que es un
    ensemble. Puede inflar la significancia al reducir varianza. La inferencia
    metodológicamente correcta es delong_per_seed_combined() abajo.
    """
    by_sub_stft = {}
    by_sub_cwt = {}
    y_truth = {}
    for r in stft_runs:
        if r is None: continue
        for sub, sc, yt in zip(r["subjects"], r["scores"], r["y_true"]):
            by_sub_stft.setdefault(sub, []).append(sc)
            y_truth[sub] = yt
    for r in cwt_runs:
        if r is None: continue
        for sub, sc in zip(r["subjects"], r["scores"]):
            by_sub_cwt.setdefault(sub, []).append(sc)
    common = sorted(set(by_sub_stft) & set(by_sub_cwt))
    a = np.array([np.median(by_sub_stft[s]) for s in common])
    b = np.array([np.median(by_sub_cwt[s]) for s in common])
    y = np.array([y_truth[s] for s in common])
    try:
        return {**delong_test(y, a, b), "n_subjects": len(common)}
    except Exception as e:
        return {"error": str(e), "n_subjects": len(common)}


def delong_per_seed_combined(stft_runs: list[dict], cwt_runs: list[dict]) -> dict:
    """DeLong dentro de cada seed (n=N sujetos por test), luego combina los
    p-values con Stouffer y Fisher. Esta es la inferencia recomendada.
    """
    from scipy.stats import combine_pvalues
    per_seed = []
    pvals = []
    for i, (ra, rb) in enumerate(zip(stft_runs, cwt_runs)):
        if ra is None or rb is None:
            continue
        by_a = dict(zip(ra["subjects"], ra["scores"]))
        by_b = dict(zip(rb["subjects"], rb["scores"]))
        truth = dict(zip(ra["subjects"], ra["y_true"]))
        common = sorted(set(by_a) & set(by_b))
        if not common:
            continue
        a = np.array([by_a[s] for s in common])
        b = np.array([by_b[s] for s in common])
        y = np.array([truth[s] for s in common])
        try:
            d = delong_test(y, a, b)
        except Exception as e:
            per_seed.append({"seed": i, "error": str(e)})
            continue
        per_seed.append({
            "seed": i,
            "n_subjects": len(common),
            "auc_a": d.get("auc_a"),
            "auc_b": d.get("auc_b"),
            "auc_diff": d.get("auc_diff"),
            "pvalue": d.get("pvalue"),
        })
        if d.get("pvalue") is not None:
            pvals.append(d["pvalue"])
    out = {"per_seed": per_seed}
    if len(pvals) >= 2:
        st = combine_pvalues(pvals, method="stouffer")
        fi = combine_pvalues(pvals, method="fisher")
        out["combined"] = {
            "stouffer": {"statistic": float(st.statistic), "pvalue": float(st.pvalue)},
            "fisher":   {"statistic": float(fi.statistic), "pvalue": float(fi.pvalue)},
            "n_seeds_combined": len(pvals),
        }
    return out


def main():
    result = {}

    print("=== CNN end-to-end ===", flush=True)
    for method in ["stft", "cwt"]:
        runs = [cnn_metrics(method, s) for s in SEEDS]
        result.setdefault("cnn", {})[method] = {
            "agg": aggregate(runs),
            "runs": [{"seed": s, **{k: r[k] for k in ("accuracy", "f1_macro", "sensitivity", "specificity", "auc")}}
                     for s, r in zip(SEEDS, runs) if r is not None],
        }

    cnn_stft = [cnn_metrics("stft", s) for s in SEEDS]
    cnn_cwt = [cnn_metrics("cwt", s) for s in SEEDS]
    result["cnn"]["wilcoxon_stft_vs_cwt"] = wilcoxon_per_seed(cnn_stft, cnn_cwt)
    result["cnn"]["delong_pooled"] = delong_pooled(cnn_stft, cnn_cwt)
    result["cnn"]["delong_per_seed"] = delong_per_seed_combined(cnn_stft, cnn_cwt)

    print("=== SVM ===", flush=True)
    for sal in ["gradcam", "vanilla"]:
        result.setdefault("svm", {})[sal] = {}
        for method in ["stft", "cwt"]:
            runs = [svm_metrics(method, s, sal) for s in SEEDS]
            result["svm"][sal][method] = {
                "agg": aggregate(runs),
                "runs": [{"seed": s, **{k: r[k] for k in ("accuracy", "f1_macro", "sensitivity", "specificity", "auc")}}
                         for s, r in zip(SEEDS, runs) if r is not None],
            }
        stft_runs = [svm_metrics("stft", s, sal) for s in SEEDS]
        cwt_runs = [svm_metrics("cwt", s, sal) for s in SEEDS]
        result["svm"][sal]["wilcoxon_stft_vs_cwt"] = wilcoxon_per_seed(stft_runs, cwt_runs)
        result["svm"][sal]["delong_pooled"] = delong_pooled(stft_runs, cwt_runs)
        result["svm"][sal]["delong_per_seed"] = delong_per_seed_combined(stft_runs, cwt_runs)

    OUT.write_text(json.dumps(result, indent=2))
    print(f"Guardado: {OUT}", flush=True)

    # Resumen legible
    print()
    print("=== Resumen ===")
    for clf_name in ("cnn",):
        print(f"\nCNN end-to-end (LOSO 65):")
        for m in ("stft", "cwt"):
            agg = result["cnn"][m]["agg"]
            print(f"  {m.upper()}: acc={agg['accuracy']['mean']:.3f}±{agg['accuracy']['std']:.3f} "
                  f"AUC={agg['auc']['mean']:.3f}±{agg['auc']['std']:.3f}")
        w = result["cnn"]["wilcoxon_stft_vs_cwt"]["per_seed"]
        if w:
            ps = [x["pvalue"] for x in w]
            print(f"  Wilcoxon p (per seed): {[f'{p:.3f}' for p in ps]}")
        d = result["cnn"]["delong_pooled"]
        if "pvalue" in d:
            print(f"  DeLong pooled: AUC_diff={d['auc_diff']:+.3f} p={d['pvalue']:.3f}")

    for sal in ("gradcam", "vanilla"):
        print(f"\nSVM con saliency {sal}:")
        for m in ("stft", "cwt"):
            agg = result["svm"][sal][m]["agg"]
            print(f"  {m.upper()}: acc={agg['accuracy']['mean']:.3f}±{agg['accuracy']['std']:.3f} "
                  f"AUC={agg['auc']['mean']:.3f}±{agg['auc']['std']:.3f}")
        w = result["svm"][sal]["wilcoxon_stft_vs_cwt"]["per_seed"]
        if w:
            ps = [x["pvalue"] for x in w]
            print(f"  Wilcoxon p (per seed): {[f'{p:.3f}' for p in ps]}")
        d = result["svm"][sal]["delong_pooled"]
        if "pvalue" in d:
            print(f"  DeLong pooled: AUC_diff={d['auc_diff']:+.3f} p={d['pvalue']:.3f}")


if __name__ == "__main__":
    main()
