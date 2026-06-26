# -*- coding: utf-8 -*-
"""
8) — оценка справедливой стоимости дополнительных портфелей.

Дополнительный портфель 1: опционы Call и Put НА фьючерс (индекс MIX)
Дополнительный портфель 2: две гипотетические облигации, эквивалентные одной
выбранной ОФЗ, но со встроенным опционом на 01.01.2026, страйк 100% номинала
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats, optimize
import config as C
import data_loader as dl
import pricing as pr

#  Модель Блэка-76 (опционы на фьючерс)
def black76(F, K, T, sigma, r, kind="C"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if kind == "C" else max(K - F, 0.0)
        return np.exp(-r * T) * intrinsic
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if kind == "C":
        return disc * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    return disc * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))


def implied_vol_black76(price, F, K, T, r, kind="C"):
    """Волатильность по наблюдаемой цене опциона"""
    f = lambda s: black76(F, K, T, s, r, kind) - price
    try:
        return optimize.brentq(f, 1e-4, 5.0, maxiter=200)
    except ValueError:
        return np.nan

#  Дополнительный портфель 1 — опционы на фьючерс MIX
def price_option_portfolio(panel: dict, expiry: pd.Timestamp = pd.Timestamp("2025-12-18")) -> dict:
    opt = dl.load_options()
    date = opt["date"]
    F = opt["future"]["settle"]
    T = (expiry - date).days / 365.25
    r = float(panel["rates"].loc[:date].iloc[-1][0.25])  
    obs = opt["options"].copy()
    obs["impl_vol"] = [implied_vol_black76(row.PRICE, F, row.STRIKE, T, r, row.TYPE)
                       for row in obs.itertuples()]
    obs["model_price"] = [black76(F, row.STRIKE, T, row.impl_vol, r, row.TYPE)
                          for row in obs.itertuples()]
    call_row = obs[obs.TYPE == "C"].iloc[0]
    sigma = call_row["impl_vol"]
    # пут-колл паритет: C - P = e^{-rT}(F - K)
    K0 = obs.STRIKE.iloc[0]
    cp = (obs[obs.TYPE == "C"].PRICE.iloc[0] - obs[obs.TYPE == "P"].PRICE.iloc[0])
    parity = np.exp(-r * T) * (F - K0)

    # Доп. портфель 1: 1 Call ITM (K<F) и 1 Put ITM (K>F).
    K_call = 265000          # < F => Call в деньгах (набл.)
    K_put = 295000           # > F => Put в деньгах (модельный)
    call_price = black76(F, K_call, T, sigma, r, "C")
    put_price = black76(F, K_put, T, sigma, r, "P")

    return {"date": date, "F": F, "T": T, "r": r, "sigma": sigma,
            "observed": obs, "parity_lhs": cp, "parity_rhs": parity,
            "portfolio1": {"call_K": K_call, "call": call_price,
                           "put_K": K_put, "put": put_price,
                           "total": call_price + put_price}}

#  Дополнительный портфель 2 — облигации со встроенными опционами
def _bond_price_vol(panel: dict, sid: str, asof: pd.Timestamp, lookback=250) -> float:
    """Историческая годовая волатильность дневной доходности грязной цены облигации."""
    dirty = pr.bond_dirty_matrix(panel)[sid]
    ret = np.log(dirty / dirty.shift(1)).loc[:asof].dropna().iloc[-lookback:]
    return float(ret.std() * np.sqrt(C.TRADING_DAYS_PER_YEAR))


def price_embedded_option_bonds(panel: dict, sid: str = "SU26224RMFS4",
                                exercise: pd.Timestamp = pd.Timestamp("2026-01-01"),
                                asof: pd.Timestamp = pd.Timestamp(C.RISK_DATE)) -> dict:
    """
    Оценивает putable и callable версии облигации sid относительно её обычной цены
    """
    meta = panel["bond_meta"].loc[sid]
    curve = panel["rates"].loc[asof]
    face = meta["FACEVALUE"]
    # 1) справедливая грязная цена
    spread = pr.calibrate_bond_spreads(panel, asof)[sid]
    plain = pr.price_bond_dirty(meta, asof, curve, spread)
    # 2) форвардная цена облигации на дату исполнения опциона
    T = (exercise - asof).days / 365.25
    r_short = pr.zero_rate(curve, np.array([T]))[0]
    # форвард = (текущая цена) / DF(T)
    sch = dl.build_coupon_schedule(meta, asof)
    cf_before = sch[sch["date"] <= exercise]
    pv_cf_before = float(np.sum(cf_before["cf"].values *
                                np.exp(-(pr.zero_rate(curve,
                                        (cf_before["date"] - asof).dt.days.values / 365.25)
                                        + spread) *
                                        (cf_before["date"] - asof).dt.days.values / 365.25)))
    fwd = (plain - pv_cf_before) / np.exp(-r_short * T)

    # 3) волатильность цены облигации
    sigma = _bond_price_vol(panel, sid, asof)
    K = face 
    # 4) опционы по модели Блэка
    def black_bond(kind):
        if T <= 0 or sigma <= 0:
            payoff = max(fwd - K, 0) if kind == "C" else max(K - fwd, 0)
            return np.exp(-r_short * T) * payoff
        d1 = (np.log(fwd / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        disc = np.exp(-r_short * T)
        if kind == "C":
            return disc * (fwd * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        return disc * (K * stats.norm.cdf(-d2) - fwd * stats.norm.cdf(-d1))

    put_val = black_bond("P")     
    call_val = black_bond("C")    

    putable = plain + put_val     
    callable_ = plain - call_val  

    return {"sid": sid, "asof": asof, "exercise": exercise, "T": T,
            "sigma_bond": sigma, "fwd": fwd, "strike": K,
            "plain": plain, "put_value": put_val, "call_value": call_val,
            "putable": putable, "callable": callable_,
            "putable_premium": put_val, "callable_discount": call_val}


if __name__ == "__main__":
    panel = dl.build_panel()
    print("Доп. портфель 1: опционы на фьючерс MIX (Блэк-76)")
    r1 = price_option_portfolio(panel)
    print(f"Дата {r1['date'].date()}, F={r1['F']:.0f}, T={r1['T']:.3f}, "
          f"r={r1['r']:.3f}, calib σ={r1['sigma']:.3f}")
    print("\nНаблюдаемые опционы (рынок vs модель по impl_vol):")
    print(r1["observed"][["SECID", "STRIKE", "TYPE", "PRICE",
                          "impl_vol", "model_price"]].round(2).to_string(index=False))
    print(f"\nПут-колл паритет: C-P={r1['parity_lhs']:.0f}  vs  "
          f"e^(-rT)(F-K)={r1['parity_rhs']:.0f}")
    p1 = r1["portfolio1"]
    print(f"\nПортфель 1: Call(K={p1['call_K']})={p1['call']:.0f}  "
          f"Put(K={p1['put_K']})={p1['put']:.0f}  Итого={p1['total']:.0f}")

    print("\nДоп. портфель 2: облигации со встроенными опционами")
    r2 = price_embedded_option_bonds(panel)
    print(f"Облигация {r2['sid']}, исполнение {r2['exercise'].date()}, "
          f"T={r2['T']:.3f}, σ_бонда={r2['sigma_bond']:.3f}, форвард={r2['fwd']:.2f}")
    print(f"Обычная (грязная):  {r2['plain']:.2f} руб.")
    print(f"Putable (оферта):   {r2['putable']:.2f} руб.  (+{r2['putable_premium']:.2f} — пут)")
    print(f"Callable (отзывная):{r2['callable']:.2f} руб.  (-{r2['callable_discount']:.2f} — колл)")