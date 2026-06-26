# -*- coding: utf-8 -*-
"""
2) выделение риск-факторов, PCA, описательная статистика.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config as C
import data_loader as dl

#  PCA
class PCAModel:
    def __init__(self, n_components: int):
        self.k = n_components

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = np.cov(Xc, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)               
        order = np.argsort(vals)[::-1]
        self.eigvals_ = vals[order]
        self.components_ = vecs[:, order[: self.k]].T  
        self.explained_variance_ratio_ = self.eigvals_[: self.k] / self.eigvals_.sum()
        self.total_var_ = self.eigvals_.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T   

    def inverse_transform(self, S: np.ndarray) -> np.ndarray:
        return S @ self.components_ + self.mean_      

#  Построение риск-факторов
def build_risk_factors(panel: dict) -> dict:
    rates = panel["rates"]                      
    dR = rates.diff().dropna()                    
    pca = PCAModel(C.N_PCA_RATE).fit(dR.values)
    scores = pca.transform(dR.values)
    rate_pcs = pd.DataFrame(scores, index=dR.index,
                            columns=[f"RatePC{i+1}" for i in range(C.N_PCA_RATE)])
    stocks = panel["stocks"]
    stock_ret = np.log(stocks / stocks.shift(1)).dropna()
    stock_ret.columns = [f"ST_{c}" for c in stock_ret.columns]
    fx = panel["indices"][C.FX_SECIDS]
    fx_ret = np.log(fx / fx.shift(1)).dropna()
    fx_ret.columns = [f"FX_{c.replace('FIXME','')}" for c in fx_ret.columns]
    factors = (rate_pcs.join(stock_ret, how="inner")
                       .join(fx_ret, how="inner")).dropna()
    levels = {
        "rates": rates,
        "stocks": stocks,
        "fx": fx,
    }
    return {"rate_changes": dR, "pca": pca, "rate_pcs": rate_pcs,
            "stock_ret": stock_ret, "fx_ret": fx_ret,
            "factors": factors, "levels": levels}

#  Описательная статистика
def descriptive_stats(rf: dict) -> pd.DataFrame:
    from scipy import stats
    from statsmodels.tsa.stattools import adfuller

    F = rf["factors"]
    rows = []
    for c in F.columns:
        x = F[c].dropna().values
        adf_stat, adf_p = adfuller(x, autolag="AIC")[:2]
        jb_stat, jb_p = stats.jarque_bera(x)[:2]
        rows.append({
            "factor": c,
            "mean": x.mean(),
            "std": x.std(ddof=1),
            "skew": stats.skew(x),
            "ex_kurtosis": stats.kurtosis(x),          
            "min": x.min(), "max": x.max(),
            "ADF_stat": adf_stat, "ADF_p": adf_p,      
            "JB_p": jb_p,                               
        })
    return pd.DataFrame(rows).set_index("factor")


def tail_heaviness(rf: dict) -> pd.DataFrame:
    from scipy import stats
    F = rf["factors"]
    rows = []
    for c in F.columns:
        x = np.abs(F[c].dropna().values)
        x = np.sort(x)[::-1]
        k = max(10, int(0.05 * len(x)))
        tail = x[:k]
        hill = 1.0 / (np.mean(np.log(tail)) - np.log(tail[-1]))   # хвостовой индекс
        rows.append({"factor": c,
                     "ex_kurtosis": stats.kurtosis(F[c].dropna().values),
                     "hill_alpha": hill})
    return pd.DataFrame(rows).set_index("factor")


def seasonality_trend(rf: dict) -> pd.DataFrame:
    from scipy import stats
    out = []
    levels = {
        "rate_5y": rf["levels"]["rates"][5.0],
        "USDRUB": rf["levels"]["fx"]["USDFIXME"],
        "EURRUB": rf["levels"]["fx"]["EURFIXME"],
    }
    for name, s in levels.items():
        s = s.dropna()
        t = np.arange(len(s))
        slope, _, r, p, _ = stats.linregress(t, s.values)
        out.append({"series": name, "trend_slope_per_day": slope,
                    "trend_R2": r**2, "trend_p": p})
    return pd.DataFrame(out).set_index("series")


def correlation_matrix(rf: dict) -> pd.DataFrame:
    return rf["factors"].corr()


if __name__ == "__main__":
    panel = dl.build_panel()
    rf = build_risk_factors(panel)
    print("Доля объяснённой дисперсии 3 компонентами кривой:",
          rf["pca"].explained_variance_ratio_.round(4),
          "| сумма:", rf["pca"].explained_variance_ratio_.sum().round(4))
    print("\nЧисло риск-факторов:", rf["factors"].shape[1],
          "| наблюдений:", rf["factors"].shape[0])
    print("\nОписательная статистика (фрагмент)")
    ds = descriptive_stats(rf)
    print(ds[["std", "skew", "ex_kurtosis", "ADF_p", "JB_p"]].round(4).to_string())
