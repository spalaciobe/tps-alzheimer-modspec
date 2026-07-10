"""Análisis de bajo coste sobre scores por sujeto ya guardados (sin re-entrenar):
  1. Late-fusion STFT + CWT-fair (promedio de scores por sujeto).
  2. TOST de equivalencia sobre ΔAUC (STFT − CWT-fair) con margen δ.
  3. Bootstrap CI por sujeto de ΔAUC (ensemble mediana entre seeds).
  4. Varianza entre seeds (nativa vs fair vs stft).

Uso: python scripts/15_fusion_tost.py
Salida: consola + results/fusion_tost_analysis.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

BASE = Path("D:/Universidad/Maestria/TPS/Proyecto/results")
SEEDS = [0, 1, 2]
DELTA = 0.05          # margen de equivalencia sobre AUC
NBOOT = 5000
RNG = np.random.RandomState(0)


def load(method: str, seed: int, sal: str = "vanilla") -> dict:
    tag = "_vanilla" if sal == "vanilla" else ""
    p = BASE / f"svm_{method}_200_seed{seed}{tag}_perfold" / "fold_results.json"
    d = json.loads(p.read_text())
    return {r["test_subject"]: (float(r["score"]), int(r["true"])) for r in d}


def auc_from(scores: dict) -> float:
    subs = sorted(scores)
    y = np.array([scores[s][1] for s in subs])
    p = np.array([scores[s][0] for s in subs])
    return roc_auc_score(y, p)


def main():
    out = {"delta": DELTA, "seeds": SEEDS}

    # ---------- por seed: AUC stft, cwt_fair, late-fusion ----------
    per_seed = {"stft": [], "cwt_fair": [], "fusion": [], "dauc_stft_minus_fair": []}
    # también guardamos scores alineados por sujeto para el ensemble
    aligned = {"stft": {}, "cwt_fair": {}, "y": {}}
    for s in SEEDS:
        st = load("stft", s)
        cf = load("cwt_fair", s)
        common = sorted(set(st) & set(cf))
        y = np.array([st[c][1] for c in common])
        p_st = np.array([st[c][0] for c in common])
        p_cf = np.array([cf[c][0] for c in common])
        p_fu = 0.5 * (p_st + p_cf)
        a_st, a_cf, a_fu = roc_auc_score(y, p_st), roc_auc_score(y, p_cf), roc_auc_score(y, p_fu)
        per_seed["stft"].append(a_st)
        per_seed["cwt_fair"].append(a_cf)
        per_seed["fusion"].append(a_fu)
        per_seed["dauc_stft_minus_fair"].append(a_st - a_cf)
        for i, c in enumerate(common):
            aligned["stft"].setdefault(c, []).append(p_st[i])
            aligned["cwt_fair"].setdefault(c, []).append(p_cf[i])
            aligned["y"][c] = y[i]

    def ms(x):
        x = np.array(x)
        return {"mean": float(x.mean()), "std": float(x.std(ddof=1))}

    out["per_seed_auc"] = {
        "stft": {**ms(per_seed["stft"]), "values": [round(v, 4) for v in per_seed["stft"]]},
        "cwt_fair": {**ms(per_seed["cwt_fair"]), "values": [round(v, 4) for v in per_seed["cwt_fair"]]},
        "late_fusion": {**ms(per_seed["fusion"]), "values": [round(v, 4) for v in per_seed["fusion"]]},
    }

    print("=== 1. Late-fusion STFT + CWT-fair (AUC, 3 seeds) ===")
    for k in ("stft", "cwt_fair", "late_fusion"):
        v = out["per_seed_auc"][k]
        print(f"  {k:12s}: {v['mean']:.3f} ± {v['std']:.3f}   {v['values']}")
    d_fu = np.array(per_seed["fusion"]) - np.array(per_seed["stft"])
    print(f"  Δ(fusion − stft) por seed: {[round(x,4) for x in d_fu]}  media {d_fu.mean():+.4f}")

    # ---------- ensemble mediana entre seeds ----------
    subs = sorted(aligned["y"])
    y_e = np.array([aligned["y"][c] for c in subs])
    st_e = np.array([np.median(aligned["stft"][c]) for c in subs])
    cf_e = np.array([np.median(aligned["cwt_fair"][c]) for c in subs])
    fu_e = 0.5 * (st_e + cf_e)
    ens = {"stft": roc_auc_score(y_e, st_e), "cwt_fair": roc_auc_score(y_e, cf_e),
           "late_fusion": roc_auc_score(y_e, fu_e)}
    out["ensemble_median_auc"] = {k: round(v, 4) for k, v in ens.items()}
    print("\n=== 2. Ensemble (mediana de scores entre seeds), AUC ===")
    for k, v in ens.items():
        print(f"  {k:12s}: {v:.3f}")

    # ---------- bootstrap por sujeto de ΔAUC (ensemble) ----------
    n = len(subs)
    boot = []
    for _ in range(NBOOT):
        idx = RNG.randint(0, n, n)
        yb = y_e[idx]
        if yb.min() == yb.max():
            continue
        boot.append(roc_auc_score(yb, st_e[idx]) - roc_auc_score(yb, cf_e[idx]))
    boot = np.array(boot)
    dauc_obs = ens["stft"] - ens["cwt_fair"]
    ci90 = (float(np.percentile(boot, 5)), float(np.percentile(boot, 95)))
    ci95 = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    out["dauc_ensemble"] = {"observed": round(dauc_obs, 4),
                            "ci90": [round(c, 4) for c in ci90],
                            "ci95": [round(c, 4) for c in ci95]}
    print("\n=== 3. ΔAUC (STFT − CWT-fair), ensemble + bootstrap por sujeto ===")
    print(f"  ΔAUC obs = {dauc_obs:+.3f} | IC90 [{ci90[0]:+.3f}, {ci90[1]:+.3f}] | IC95 [{ci95[0]:+.3f}, {ci95[1]:+.3f}]")

    # ---------- TOST de equivalencia ----------
    # (a) TOST sobre las 3 ΔAUC por seed (t de una muestra vs ±δ)
    d_seed = np.array(per_seed["dauc_stft_minus_fair"])
    mean_d, sd_d, nseed = d_seed.mean(), d_seed.std(ddof=1), len(d_seed)
    se_d = sd_d / np.sqrt(nseed)
    # H0_upper: d >= +δ  → t = (mean - δ)/se, one-sided lower
    t_low = (mean_d - DELTA) / se_d
    t_up = (mean_d + DELTA) / se_d
    p_upper = stats.t.cdf(t_low, nseed - 1)      # P(T <= t_low): prueba que d < δ
    p_lower = stats.t.sf(t_up, nseed - 1)        # P(T >= t_up): prueba que d > -δ
    p_tost = max(p_upper, p_lower)
    # (b) equivalencia vía IC bootstrap: equivalente si IC90 ⊂ [-δ, δ]
    equiv_ci = (ci90[0] > -DELTA) and (ci90[1] < DELTA)
    out["tost"] = {
        "delta": DELTA,
        "dauc_per_seed": [round(x, 4) for x in d_seed],
        "mean": round(float(mean_d), 4), "sd": round(float(sd_d), 4),
        "p_tost_seedlevel": round(float(p_tost), 4),
        "equivalente_seedlevel": bool(p_tost < 0.05),
        "equivalente_por_IC90_bootstrap": bool(equiv_ci),
    }
    print(f"\n=== 4. TOST de equivalencia (δ = ±{DELTA} AUC) ===")
    print(f"  ΔAUC por seed: {[round(x,4) for x in d_seed]}  media {mean_d:+.4f} ± {sd_d:.4f}")
    print(f"  TOST seed-level p = {p_tost:.4f}  → {'EQUIVALENTE' if p_tost<0.05 else 'NO concluyente (equivalencia no demostrada)'} al 5%")
    print(f"  Equivalencia por IC90 bootstrap ⊂ [±{DELTA}]: {'SÍ' if equiv_ci else 'NO'} (IC90 [{ci90[0]:+.3f},{ci90[1]:+.3f}])")
    # margen mínimo δ que haría equivalente por IC90
    delta_min = max(abs(ci90[0]), abs(ci90[1]))
    out["tost"]["delta_min_para_equivalencia_IC90"] = round(float(delta_min), 4)
    print(f"  δ mínimo para declarar equivalencia (IC90): {delta_min:.3f} AUC")

    # ---------- varianza entre seeds (incluye CWT nativa) ----------
    def auc_series(method):
        vals = []
        for s in SEEDS:
            sc = load(method, s)
            vals.append(auc_from(sc))
        return np.array(vals)
    var = {}
    for m in ("stft", "cwt", "cwt_fair"):
        try:
            a = auc_series(m)
            var[m] = {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4)}
        except Exception as e:
            var[m] = {"error": str(e)}
    out["varianza_entre_seeds_auc"] = var
    print("\n=== 5. Varianza entre seeds (AUC vainilla) ===")
    for m in ("stft", "cwt", "cwt_fair"):
        if "sd" in var[m]:
            print(f"  {m:9s}: {var[m]['mean']:.3f} ± {var[m]['sd']:.3f}")

    (BASE / "fusion_tost_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nGuardado: {BASE / 'fusion_tost_analysis.json'}")


if __name__ == "__main__":
    main()
