"""Análisis estratificado por género.

Motivación: el dataset ds004504 tiene sesgo de género (chi2 p=0.039,
24F/12M en AD vs 11F/18M en HC). ¿La accuracy del mejor modelo varía
sustantivamente entre hombres y mujeres?

Para la mejor configuración (SVM vainilla STFT, 3 seeds), calcula
accuracy/AUC por género y verifica si las diferencias son significativas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto")
OUT = BASE / "results/post_experiments"


def load_gender_map() -> dict[str, str]:
    df = pd.read_csv(BASE / "data/raw/ds004504/participants.tsv", sep="\t")
    df = df.rename(columns={"participant_id": "sid", "Gender": "g", "Group": "grp"})
    return dict(zip(df["sid"], df["g"]))


def metrics_per_group(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    y_true = np.array([r["true"] for r in rows])
    y_pred = np.array([r["pred"] for r in rows])
    y_score = np.array([r["score"] for r in rows])
    correct = int((y_true == y_pred).sum())
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    sens = tp / max(n_pos, 1)
    spec = tn / max(n_neg, 1)
    acc = correct / len(y_true)
    auc = None
    if 0 < n_pos < len(y_true):
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    return {"n": len(y_true), "n_AD": n_pos, "n_HC": n_neg,
            "accuracy": acc, "sensitivity": sens, "specificity": spec, "auc": auc}


def main():
    gender_map = load_gender_map()
    print("=== Análisis estratificado por género ===", flush=True)
    print("Mejor modelo: SVM vainilla STFT (3 seeds)\n")

    # Cargar todos los seeds y agregar por sujeto (mediana)
    by_subject_score = {}
    by_subject_true = {}
    by_subject_pred_votes = {}
    for s in (0, 1, 2):
        p = BASE / f"results/svm_stft_200_seed{s}_vanilla_perfold/fold_results.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            sid = r["test_subject"]
            by_subject_score.setdefault(sid, []).append(r["score"])
            by_subject_true[sid] = r["true"]
            by_subject_pred_votes.setdefault(sid, []).append(r["pred"])

    # Predicción agregada: mediana de scores → threshold 0.5
    rows = []
    for sid, scores in by_subject_score.items():
        agg = float(np.median(scores))
        pred = int(agg >= 0.5)
        rows.append({
            "subject": sid,
            "gender": gender_map.get(sid, "?"),
            "true": by_subject_true[sid],
            "pred": pred,
            "score": agg,
        })

    print(f"Total sujetos agregados: {len(rows)}")
    print()

    # Por género
    result = {"overall": metrics_per_group(rows)}
    for g in ("F", "M"):
        rows_g = [r for r in rows if r["gender"] == g]
        result[g] = metrics_per_group(rows_g)
        print(f"  Género {g} (n={result[g]['n']}, AD={result[g]['n_AD']}, HC={result[g]['n_HC']}):")
        print(f"    acc={result[g]['accuracy']:.3f}  sens={result[g]['sensitivity']:.3f}  "
              f"spec={result[g]['specificity']:.3f}  auc={result[g]['auc']:.3f}")

    # Test estadístico: ¿la acc es diferente entre F y M?
    f_rows = [r for r in rows if r["gender"] == "F"]
    m_rows = [r for r in rows if r["gender"] == "M"]
    f_correct = np.array([int(r["pred"] == r["true"]) for r in f_rows])
    m_correct = np.array([int(r["pred"] == r["true"]) for r in m_rows])
    # Fisher exact sobre tabla 2x2
    f_right, f_wrong = f_correct.sum(), len(f_correct) - f_correct.sum()
    m_right, m_wrong = m_correct.sum(), len(m_correct) - m_correct.sum()
    odds, p_fisher = stats.fisher_exact([[f_right, f_wrong], [m_right, m_wrong]])
    result["test_acc_F_vs_M"] = {
        "fisher_exact_p": float(p_fisher),
        "odds": float(odds),
        "F_correct": int(f_right), "F_total": len(f_correct),
        "M_correct": int(m_right), "M_total": len(m_correct),
    }
    print()
    print(f"Fisher exact F vs M (correctly classified): p={p_fisher:.4f}")

    # Comparación con la métrica global
    print()
    print(f"Overall: acc={result['overall']['accuracy']:.3f}  AUC={result['overall']['auc']:.3f}")
    print()
    if p_fisher < 0.05:
        print("CONCLUSIÓN: hay diferencia significativa en accuracy entre géneros (posible sesgo).")
    else:
        print("CONCLUSIÓN: no hay diferencia significativa en accuracy entre géneros (modelo robusto al sesgo).")

    (OUT / "gender_stratified.json").write_text(json.dumps(result, indent=2))
    print(f"\nGuardado: {OUT / 'gender_stratified.json'}")


if __name__ == "__main__":
    main()
