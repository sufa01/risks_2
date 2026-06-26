# -*- coding: utf-8 -*-
"""
4) оценка справедливой стоимости инструментов портфеля
в зависимости от риск-факторов, плюс проверка точности.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
import data_loader as dl

# Интерполяция кривой и дисконт-факторы
def zero_rate(curve: pd.Series, t_years: np.ndarray) -> np.ndarray:
    """
    Линейная интерполяция zero-ставки по сроку
    """
    xs = np.asarray(curve.index, dtype=float)
    ys = np.asarray(curve.values, dtype=float)
    t = np.asarray(t_years, dtype=float)
    return np.interp(t, xs, ys, left=ys[0], right=ys[-1])


def discount_factor(curve: pd.Series, t_years: np.ndarray) -> np.ndarray:
    """Дисконт-фактор при непрерывном начислении"""
    r = zero_rate(curve, t_years)
    t = np.asarray(t_years, dtype=float)
    return np.exp(-r * t)

# Облигации
def price_bond_dirty(meta_row: pd.Series, asof: pd.Timestamp,
                     curve: pd.Series, spread: float = 0.0) -> float:
    sch = dl.build_coupon_schedule(meta_row, asof)
    if sch.empty:
        return np.nan
    t = (sch["date"] - asof).dt.days.values / 365.25
    r = zero_rate(curve, t) + spread
    df = np.exp(-r * t)
    return float(np.sum(sch["cf"].values * df))


def calibrate_bond_spreads(panel: dict, asof: pd.Timestamp) -> pd.Series:
    from scipy.optimize import brentq
    curve = panel["rates"].loc[asof]
    meta = panel["bond_meta"]
    spreads = {}
    for sid in meta.index:
        clean = panel["bond_clean"].at[asof, sid]           
        acc = panel["bond_accint"].at[asof, sid]            
        face = meta.at[sid, "FACEVALUE"]
        mkt_dirty = clean / 100.0 * face + acc
        f = lambda s: price_bond_dirty(meta.loc[sid], asof, curve, s) - mkt_dirty
        try:
            spreads[sid] = brentq(f, -0.20, 0.20, maxiter=200)
        except ValueError:
            spreads[sid] = 0.0
    return pd.Series(spreads)

def bond_market_dirty(panel: dict, asof: pd.Timestamp) -> pd.Series:
    meta = panel["bond_meta"]
    clean = panel["bond_clean"].loc[asof]
    acc = panel["bond_accint"].loc[asof]
    return clean / 100.0 * meta["FACEVALUE"] + acc


def bond_dirty_matrix(panel: dict) -> pd.DataFrame:
    if "_bond_dirty" in panel:
        return panel["_bond_dirty"]
    meta = panel["bond_meta"]
    dates = panel["dates"]
    mkt = (panel["bond_clean"] / 100.0 * meta["FACEVALUE"] + panel["bond_accint"])
    out = mkt.copy()
    # модельная цена по КБД там, где рынок отсутствует
    for sid in meta.index:
        missing = out[sid].isna()
        if missing.any():
            for d in dates[missing.values]:
                out.at[d, sid] = price_bond_dirty(meta.loc[sid],
                                                  d, panel["rates"].loc[d], 0.0)
    panel["_bond_dirty"] = out
    return out

# Проверка точности модели облигаций
def bond_pricing_accuracy(panel: dict, dates=None) -> pd.DataFrame:
    if dates is None:
        dates = panel["dates"][::63] 
    meta = panel["bond_meta"]
    rows = []
    for d in dates:
        curve = panel["rates"].loc[d]
        for sid in meta.index:
            model = price_bond_dirty(meta.loc[sid], d, curve, 0.0)
            mkt = bond_market_dirty(panel, d)[sid]
            face = meta.at[sid, "FACEVALUE"]
            rows.append({"date": d, "SECID": sid,
                         "model": model, "market": mkt,
                         "err_rub": model - mkt,
                         "err_pct_face": (model - mkt) / face * 100})
    return pd.DataFrame(rows)

# Стоимость позиций портфеля
def portfolio_units(panel: dict, asof: pd.Timestamp) -> dict:
    units = {}
    dirty = bond_dirty_matrix(panel).loc[asof]
    for sid in C.BOND_SECIDS:
        units[("bond", sid)] = C.NOTIONAL_PER_BOND / dirty[sid]
    px = panel["stocks"].loc[asof]
    for t in C.STOCK_TICKERS:
        units[("stock", t)] = C.NOTIONAL_PER_STOCK / px[t]
    fx = panel["indices"].loc[asof]
    for f in C.FX_SECIDS:
        units[("fx", f)] = C.NOTIONAL_PER_FX / fx[f]
    return units


if __name__ == "__main__":
    panel = dl.build_panel()
    asof = pd.Timestamp(C.RISK_DATE)
    print("Точность ценообразования облигаций по КБД (без спреда)")
    acc = bond_pricing_accuracy(panel, dates=[asof])
    print(acc[["SECID", "model", "market", "err_rub", "err_pct_face"]].to_string(index=False))
    print("\nMAE по %номинала:", acc["err_pct_face"].abs().mean().round(3))
    print("\nКалиброванные z-спреды (грязная цена точно = рынок)")
    sp = calibrate_bond_spreads(panel, asof)
    print((sp * 100).round(3).rename("z-spread, %").to_string())