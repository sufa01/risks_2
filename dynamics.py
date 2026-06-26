# -*- coding: utf-8 -*-
"""
3) стохастические модели динамики риск-факторов и оценка
параметров методом максимального правдоподобия (MLE).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import config as C

#  MLE маргинальных распределений
def fit_student_t(x: np.ndarray) -> dict:
    """
    MLE распределения Стьюдента t(df, loc, scale) с нижним порогом df
    для конечной дисперсии и устойчивого Монте-Карло
    """
    df, loc, scale = stats.t.fit(x)
    if df < C.T_DF_FLOOR:
        df = C.T_DF_FLOOR
        loc, scale = stats.t.fit(x, fdf=df)[1:]
    ll = np.sum(stats.t.logpdf(x, df, loc, scale))
    aic = 2 * 3 - 2 * ll
    return {"dist": "t", "df": df, "loc": loc, "scale": scale,
            "loglik": ll, "aic": aic}

def fit_normal(x: np.ndarray) -> dict:
    """MLE нормального распределения"""
    loc, scale = stats.norm.fit(x)
    ll = np.sum(stats.norm.logpdf(x, loc, scale))
    aic = 2 * 2 - 2 * ll
    return {"dist": "norm", "loc": loc, "scale": scale,
            "loglik": ll, "aic": aic}


def fit_marginals(factors: pd.DataFrame) -> dict:
    """
    Оценивает MLE-параметры t-распределения для каждого риск-фактора
    """
    out = {}
    for c in factors.columns:
        x = factors[c].dropna().values
        out[c] = {"t": fit_student_t(x), "norm": fit_normal(x)}
    return out


def marginals_table(marg: dict) -> pd.DataFrame:
    """Сводная таблица параметров маргиналов"""
    rows = []
    for f, d in marg.items():
        t, n = d["t"], d["norm"]
        rows.append({"factor": f, "t_df": t["df"], "t_scale": t["scale"],
                     "t_loc": t["loc"], "AIC_t": t["aic"], "AIC_norm": n["aic"],
                     "t_better": t["aic"] < n["aic"]})
    return pd.DataFrame(rows).set_index("factor")

#  Гауссова копула: корреляция по нормальным скорам
def gaussian_copula_corr(factors: pd.DataFrame, marg: dict) -> np.ndarray:
    """
    Корреляционная матрица гауссовой копулы
    """
    cols = list(factors.columns)
    Z = np.empty((len(factors), len(cols)))
    for j, c in enumerate(cols):
        p = marg[c]["t"]
        u = stats.t.cdf(factors[c].values, p["df"], p["loc"], p["scale"])
        u = np.clip(u, 1e-6, 1 - 1e-6)
        Z[:, j] = stats.norm.ppf(u)
    R = np.corrcoef(Z, rowvar=False)
    R = _nearest_pd(R)
    return R


def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """Ближайшая положительно определённая матрица (симметризация + сдвиг λ)"""
    B = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(B)
    vals = np.clip(vals, 1e-8, None)
    B = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(B))
    B = B / np.outer(d, d)
    return B

#  Симуляция риск-факторов на горизонте
def simulate_factor_paths(factors: pd.DataFrame, marg: dict, R: np.ndarray,
                          n_sim: int, horizon: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Возвращает инновации риск-факторов
    """
    cols = list(factors.columns)
    m = len(cols)
    L = np.linalg.cholesky(R)
    out = np.empty((n_sim, horizon, m))
    for d in range(horizon):
        Z = rng.standard_normal((n_sim, m)) @ L.T
        U = np.clip(stats.norm.cdf(Z), 1e-6, 1 - 1e-6)
        for j, c in enumerate(cols):
            p = marg[c]["t"]
            out[:, d, j] = stats.t.ppf(U[:, j], p["df"], p["loc"], p["scale"])
    return out


def simulate_factor_innovations(factors: pd.DataFrame, marg: dict, R: np.ndarray,
                                 n_sim: int, horizon: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Возвращает массив (n_sim x n_factors) суммарных инноваций риск-факторов
    """
    cols = list(factors.columns)
    m = len(cols)
    L = np.linalg.cholesky(R)
    total = np.zeros((n_sim, m))
    for _ in range(horizon):
        Z = rng.standard_normal((n_sim, m)) @ L.T
        U = stats.norm.cdf(Z)
        U = np.clip(U, 1e-6, 1 - 1e-6)
        X = np.empty((n_sim, m))
        for j, c in enumerate(cols):
            p = marg[c]["t"]
            X[:, j] = stats.t.ppf(U[:, j], p["df"], p["loc"], p["scale"])
        total += X
    return total

#  Опционально: GARCH(1,1)-t (альтернатива, для обсуждения)
def fit_garch(x: np.ndarray):
    """GARCH(1,1) со стьюдентовскими инновациями"""
    try:
        from arch import arch_model
    except Exception:
        return None
    am = arch_model(x * 100, vol="Garch", p=1, q=1, dist="t", mean="Constant")
    res = am.fit(disp="off")
    return res


if __name__ == "__main__":
    import data_loader as dl, risk_factors as rfm
    panel = dl.build_panel()
    rf = rfm.build_risk_factors(panel)
    marg = fit_marginals(rf["factors"])
    tbl = marginals_table(marg)
    print("=== MLE маргиналов (t vs Normal) ===")
    print(tbl[["t_df", "t_scale", "AIC_t", "AIC_norm", "t_better"]].round(3).to_string())
    print("\nДля скольких факторов t лучше нормали по AIC:",
          int(tbl["t_better"].sum()), "из", len(tbl))

    R = gaussian_copula_corr(rf["factors"], marg)
    print("\nКопула-корреляция USD/EUR:", round(R[-2, -1], 3))

    rng = C.get_rng()
    sim = simulate_factor_innovations(rf["factors"], marg, R,
                                      n_sim=10000, horizon=1, rng=rng)
    print("Симуляция OK, форма:", sim.shape,
          "| ст.откл. факторов (выборка) близко к данным:",
          np.allclose(sim.std(0), rf["factors"].std().values, rtol=0.3))