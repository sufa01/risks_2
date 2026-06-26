# -*- coding: utf-8 -*-
"""
backtesting.py
6) — количественная валидация (бэктестинг) оценки VaR по всему портфелю
       и по 3 подпортфелям (акции, облигации, валюта) на каждый торговый
       день 2025 г.: расчёт VaR, подсчёт пробоев, проверка корректности
7) — статистические тесты бэктеста:
       1) Kupiec (1995)            — тест безусловного покрытия (POF)
       2) Christoffersen (1998)    — тесты независимости и условного покрытия
       3) Engle & Manganelli (2004) — Dynamic Quantile (DQ) тест
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import config as C
import data_loader as dl
import pricing as pr
import risk_factors as rfm
import dynamics as dyn
import risk_measures as rm

def realized_daily_pnl(panel: dict) -> pd.DataFrame:
    """
    Дневной P&L позиции = N_i * (цена_t / цена_{t-1} - 1).
    """
    dirty = pr.bond_dirty_matrix(panel)
    ret_b = dirty.pct_change()
    ret_s = panel["stocks"].pct_change()
    fx = panel["indices"][C.FX_SECIDS]
    ret_f = fx.pct_change()

    pnl_b = ret_b[C.BOND_SECIDS].mul(C.NOTIONAL_PER_BOND).sum(axis=1)
    pnl_s = ret_s[C.STOCK_TICKERS].mul(C.NOTIONAL_PER_STOCK).sum(axis=1)
    pnl_f = ret_f[C.FX_SECIDS].mul(C.NOTIONAL_PER_FX).sum(axis=1)
    out = pd.DataFrame({"bonds": pnl_b, "stocks": pnl_s, "fx": pnl_f})
    out["total"] = out.sum(axis=1)
    return out.dropna()

def forecast_var_1d(panel: dict, rf_full: dict, marg: dict, R: np.ndarray,
                    asof: pd.Timestamp, n_sim: int, rng,
                    level: float = C.VAR_LEVEL) -> dict:
    """
    Возвращает 1-дневный VaR
    """
    rep = rm.build_repricer(panel, rf_full, asof)
    paths = dyn.simulate_factor_paths(rf_full["factors"], marg, R, n_sim, 1, rng)
    pnl = rm.simulate_pnl(rep, paths)
    return {k: rm.value_at_risk(pnl[k], level) for k in pnl}


def run_backtest(panel: dict, year: int = C.BACKTEST_YEAR,
                 n_sim: int = C.N_SIM_BACKTEST,
                 level: float = C.VAR_LEVEL, verbose: bool = True) -> pd.DataFrame:
    """
    6) на каждый торговый день года рассчитывает VaR (прогноз с
    предыдущего дня)
    """
    rng = C.get_rng()
    realized = realized_daily_pnl(panel)
    dates = panel["dates"]
    test_days = dates[(dates.year == year)]
    rf_full = rfm.build_risk_factors(panel)
    rows = []
    marg = R = None
    last_refit = -10 ** 9
    for i, d in enumerate(test_days):
        prev = dates[dates < d][-1]
        F_win = rf_full["factors"].loc[:prev]
        if len(F_win) < 250:
            continue
        if (i - last_refit) >= C.REFIT_EVERY or marg is None:
            rf_win = dict(rf_full); rf_win["factors"] = F_win
            marg = dyn.fit_marginals(F_win)
            R = dyn.gaussian_copula_corr(F_win, marg)
            last_refit = i
        rf_for_pred = dict(rf_full); rf_for_pred["factors"] = F_win
        var = forecast_var_1d(panel, rf_for_pred, marg, R, prev, n_sim, rng, level)
        rzd = realized.loc[d]
        row = {"date": d}
        for k in ["total", "bonds", "stocks", "fx"]:
            row[f"VaR_{k}"] = var[k]
            row[f"PnL_{k}"] = rzd[k]
            row[f"breach_{k}"] = int(rzd[k] < -var[k])
        rows.append(row)
        if verbose and (i % 30 == 0):
            print(f"  ...{d.date()} обработано ({i+1}/{len(test_days)})")
    return pd.DataFrame(rows).set_index("date")

#  Сводка по пробоям
def breach_summary(bt: pd.DataFrame, level: float = C.VAR_LEVEL) -> pd.DataFrame:
    n = len(bt)
    exp_rate = 1 - level
    rows = []
    for k in ["total", "bonds", "stocks", "fx"]:
        x = bt[f"breach_{k}"].sum()
        rows.append({"sub": k, "n_days": n, "breaches": int(x),
                     "breach_rate": x / n, "expected_rate": exp_rate,
                     "expected_breaches": exp_rate * n})
    return pd.DataFrame(rows).set_index("sub")

#  7 — статистические тесты
def kupiec_pof(breaches: np.ndarray, level: float = C.VAR_LEVEL) -> dict:
    """
    Тест Купика 
    """
    x = np.asarray(breaches).astype(int)
    n = len(x); n1 = x.sum(); n0 = n - n1
    p = 1 - level
    pi = n1 / n if n > 0 else 0.0
    if n1 == 0:
        lr = -2 * (n0 * np.log(1 - p))
    elif n1 == n:
        lr = -2 * (n1 * np.log(p))
    else:
        ll0 = n0 * np.log(1 - p) + n1 * np.log(p)
        ll1 = n0 * np.log(1 - pi) + n1 * np.log(pi)
        lr = -2 * (ll0 - ll1)
    pval = 1 - stats.chi2.cdf(lr, 1)
    return {"test": "Kupiec POF", "LR": lr, "df": 1, "p_value": pval,
            "reject_H0": pval < 0.05, "n_breach": int(n1), "expected": p * n}


def christoffersen(breaches: np.ndarray, level: float = C.VAR_LEVEL) -> dict:
    """
    Тест Кристофферсена
    """
    x = np.asarray(breaches).astype(int)
    # переходы 00,01,10,11
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(x[:-1], x[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if x.size > 1 else 0.0

    def safe(a, b):
        return a * np.log(b) if (a > 0 and b > 0) else 0.0

    ll_ind = safe(n00 + n10, 1 - pi) + safe(n01 + n11, pi)
    ll_dep = (safe(n00, 1 - pi01) + safe(n01, pi01)
              + safe(n10, 1 - pi11) + safe(n11, pi11))
    lr_ind = -2 * (ll_ind - ll_dep)
    p_ind = 1 - stats.chi2.cdf(lr_ind, 1)
    pof = kupiec_pof(x, level)
    lr_cc = pof["LR"] + lr_ind
    p_cc = 1 - stats.chi2.cdf(lr_cc, 2)
    return {"test": "Christoffersen",
            "LR_ind": lr_ind, "p_ind": p_ind,
            "LR_cc": lr_cc, "p_cc": p_cc,
            "reject_ind": p_ind < 0.05, "reject_cc": p_cc < 0.05}


def dq_test(breaches: np.ndarray, var: np.ndarray, level: float = C.VAR_LEVEL,
            lags: int = 4) -> dict:
    """
    DQ тест Engle & Manganelli 
    """
    I = np.asarray(breaches).astype(float)
    p = 1 - level
    Hit = I - p
    n = len(Hit)
    if n <= lags + 2:
        return {"test": "DQ (Engle-Manganelli)", "stat": np.nan, "p_value": np.nan}
    X = [np.ones(n - lags)]
    for L in range(1, lags + 1):
        X.append(Hit[lags - L: n - L])
    v = np.asarray(var, float)[lags:]
    X.append((v - v.mean()) / (v.std() + 1e-12))
    X = np.column_stack(X)
    y = Hit[lags:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    stat = float(beta @ (X.T @ X) @ beta / (p * (1 - p)))
    k = X.shape[1]
    pval = 1 - stats.chi2.cdf(stat, k)
    return {"test": "DQ (Engle-Manganelli)", "stat": stat, "df": k,
            "p_value": pval, "reject_H0": pval < 0.05}


def run_all_tests(bt: pd.DataFrame, subs=("total", "stocks", "bonds", "fx"),
                  level: float = C.VAR_LEVEL) -> pd.DataFrame:
    """Сводная таблица тестов"""
    rows = []
    for k in subs:
        br = bt[f"breach_{k}"].values
        var = bt[f"VaR_{k}"].values
        kp = kupiec_pof(br, level)
        ch = christoffersen(br, level)
        dq = dq_test(br, var, level)
        rows.append({
            "sub": k, "breaches": int(br.sum()), "expected": round(kp["expected"], 1),
            "Kupiec_p": round(kp["p_value"], 4), "Kupiec_ok": not kp["reject_H0"],
            "Christ_ind_p": round(ch["p_ind"], 4), "Christ_cc_p": round(ch["p_cc"], 4),
            "Christ_cc_ok": not ch["reject_cc"],
            "DQ_p": round(dq["p_value"], 4), "DQ_ok": not dq["reject_H0"],
        })
    return pd.DataFrame(rows).set_index("sub")


if __name__ == "__main__":
    panel = dl.build_panel()
    print("Запуск бэктеста VaR за", C.BACKTEST_YEAR, "г. ...")
    bt = run_backtest(panel, verbose=True)
    print("\n6: Сводка по пробоям")
    print(breach_summary(bt).round(3).to_string())
    print("\n7: Статистические тесты")
    print(run_all_tests(bt).to_string())
    bt.to_csv(C.OUT_DIR + "/backtest_2025.csv")