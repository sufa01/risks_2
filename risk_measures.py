# -*- coding: utf-8 -*-
"""
5) оценка рыночного риска портфеля методом Монте-Карло
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
import data_loader as dl
import pricing as pr

#  Подготовка статических данных для переоценки на дату
def _interp_matrix(tenors: np.ndarray, t_cf: np.ndarray) -> np.ndarray:
    tn = np.asarray(tenors, float)
    M = np.zeros((len(t_cf), len(tn)))
    for i, t in enumerate(t_cf):
        if t <= tn[0]:
            M[i, 0] = 1.0
        elif t >= tn[-1]:
            M[i, -1] = 1.0
        else:
            k = np.searchsorted(tn, t) - 1
            w = (t - tn[k]) / (tn[k + 1] - tn[k])
            M[i, k] = 1 - w
            M[i, k + 1] = w
    return M


def build_repricer(panel: dict, rf: dict, asof: pd.Timestamp) -> dict:
    meta = panel["bond_meta"]
    curve0 = panel["rates"].loc[asof]
    tenors = np.array(C.CURVE_TENORS, float)
    spreads = pr.calibrate_bond_spreads(panel, asof)

    bonds = {}
    for sid in C.BOND_SECIDS:
        sch = dl.build_coupon_schedule(meta.loc[sid], asof)
        t_cf = (sch["date"] - asof).dt.days.values / 365.25
        cf = sch["cf"].values
        r0_cf = pr.zero_rate(curve0, t_cf)
        M = _interp_matrix(tenors, t_cf)
        base_dirty = float(np.sum(cf * np.exp(-(r0_cf + spreads[sid]) * t_cf)))
        bonds[sid] = {"t_cf": t_cf, "cf": cf, "r0_cf": r0_cf,
                      "spread": spreads[sid], "M": M, "base": base_dirty}

    units = pr.portfolio_units(panel, asof)
    stock_px0 = panel["stocks"].loc[asof]
    fx0 = panel["indices"].loc[asof][C.FX_SECIDS]

    # порядок факторов и индексы групп
    fcols = list(rf["factors"].columns)
    idx_pc = [fcols.index(f"RatePC{i+1}") for i in range(C.N_PCA_RATE)]
    idx_st = {t: fcols.index(f"ST_{t}") for t in C.STOCK_TICKERS}
    idx_fx = {f: fcols.index(f"FX_{f.replace('FIXME','')}") for f in C.FX_SECIDS}

    return {"asof": asof, "bonds": bonds, "units": units,
            "stock_px0": stock_px0, "fx0": fx0,
            "pca": rf["pca"], "fcols": fcols,
            "idx_pc": idx_pc, "idx_st": idx_st, "idx_fx": idx_fx,
            "spreads": spreads}


def _bond_price_from_shift(b: dict, shift12: np.ndarray) -> np.ndarray:
    shift_cf = shift12 @ b["M"].T                  
    rate_cf = b["r0_cf"][None, :] + shift_cf + b["spread"]
    price = (b["cf"][None, :] * np.exp(-rate_cf * b["t_cf"][None, :])).sum(axis=1)
    return price

#  P&L портфеля по сымитированным траекториям факторов
def simulate_pnl(rep: dict, paths: np.ndarray) -> dict:
    """
    Считает выборку P&L портфеля и 3 подпортфелей по подённым инновациям
    """
    n_sim, horizon, _ = paths.shape
    pca = rep["pca"]; comp = pca.components_; mean = pca.mean_

    pnl_b = np.zeros(n_sim)
    pnl_s = np.zeros(n_sim)
    pnl_f = np.zeros(n_sim)

    # облигации
    cum_shift = np.zeros((n_sim, len(C.CURVE_TENORS)))
    # базовые цены (shift=0)
    prev_price = {sid: np.full(n_sim, rep["bonds"][sid]["base"])
                  for sid in C.BOND_SECIDS}
    for d in range(horizon):
        pc_scores = paths[:, d, rep["idx_pc"]]            
        dR12 = pc_scores @ comp + mean                    
        cum_shift = cum_shift + dR12
        for sid in C.BOND_SECIDS:
            b = rep["bonds"][sid]
            price_d = _bond_price_from_shift(b, cum_shift)
            ret = price_d / prev_price[sid] - 1.0
            pnl_b += C.NOTIONAL_PER_BOND * ret             
            prev_price[sid] = price_d

    # акции: суточная ребалансировка
    for t in C.STOCK_TICKERS:
        j = rep["idx_st"][t]
        daily_ret = np.exp(paths[:, :, j]) - 1.0          
        pnl_s += C.NOTIONAL_PER_STOCK * daily_ret.sum(axis=1)

    # валюта: суточная ребалансировка
    for f in C.FX_SECIDS:
        j = rep["idx_fx"][f]
        daily_ret = np.exp(paths[:, :, j]) - 1.0
        pnl_f += C.NOTIONAL_PER_FX * daily_ret.sum(axis=1)

    total = pnl_b + pnl_s + pnl_f
    return {"total": total, "bonds": pnl_b, "stocks": pnl_s, "fx": pnl_f}

#  Меры риска
def value_at_risk(pnl: np.ndarray, level: float = C.VAR_LEVEL) -> float:
    return float(-np.quantile(pnl, 1 - level))

def expected_shortfall(pnl: np.ndarray, level: float = C.ES_LEVEL) -> float:
    q = np.quantile(pnl, 1 - level)
    tail = pnl[pnl <= q]
    return float(-tail.mean()) if len(tail) else float(-q)


def risk_report(rep: dict, factors, marg, R, horizon: int, n_sim: int,
                rng) -> pd.DataFrame:
    import dynamics as dyn
    paths = dyn.simulate_factor_paths(factors, marg, R, n_sim, horizon, rng)
    pnl = simulate_pnl(rep, paths)
    rows = []
    for name in ["total", "bonds", "stocks", "fx"]:
        rows.append({"sub": name, "horizon": horizon,
                     "VaR_99": value_at_risk(pnl[name], C.VAR_LEVEL),
                     "ES_975": expected_shortfall(pnl[name], C.ES_LEVEL),
                     "mean_PnL": pnl[name].mean(),
                     "std_PnL": pnl[name].std()})
    return pd.DataFrame(rows).set_index("sub")

if __name__ == "__main__":
    import risk_factors as rfm, dynamics as dyn
    panel = dl.build_panel()
    rf = rfm.build_risk_factors(panel)
    marg = dyn.fit_marginals(rf["factors"])
    R = dyn.gaussian_copula_corr(rf["factors"], marg)
    asof = pd.Timestamp(C.RISK_DATE)
    rep = build_repricer(panel, rf, asof)
    rng = C.get_rng()
    for h in C.HORIZONS:
        print(f"\nМеры риска, горизонт {h} дн., дата {C.RISK_DATE} "
              f"(VaR 99%, ES 97.5%), руб.")
        rr = risk_report(rep, rf["factors"], marg, R, h, C.N_SIM, rng)
        print(rr.round(0).to_string())
        tot = rr.loc["total"]
        print(f"VaR в % портфеля: {tot['VaR_99']/C.TOTAL_NOTIONAL*100:.2f}% | "
              f"ES: {tot['ES_975']/C.TOTAL_NOTIONAL*100:.2f}%")