# -*- coding: utf-8 -*-
"""
  fig_rates_history     — история кривой и 3 главных компонент
  fig_pca_loadings      — нагрузки PC (уровень/наклон/кривизна) + explained variance
  fig_factor_history    — история инноваций риск-факторов (выборка)
  fig_corr_heatmap      — тепловая карта корреляций риск-факторов
  fig_tails             — QQ-графики и гистограммы хвостов (t vs Normal)
  fig_levels            — уровни курсов и индексов (тренд)
  fig_backtest          — P&L против -VaR с пробоями (по портфелю и подпортфелям)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config as C

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": True})

def fig_rates_history(panel, rf, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    r = panel["rates"]
    for t in [0.25, 2.0, 10.0, 30.0]:
        ax[0].plot(r.index, r[t] * 100, label=f"{t:g} лет", lw=0.9)
    ax[0].set_title("Кривая бескупонной доходности (КБД), %")
    ax[0].legend(fontsize=7)
    pcs = rf["rate_pcs"]
    for c in pcs.columns:
        ax[1].plot(pcs.index, pcs[c].cumsum(), label=c, lw=0.8)
    ax[1].set_title("Накопленные главные компоненты ставок")
    ax[1].legend(fontsize=7)
    fig.savefig(path); plt.close(fig)

def fig_pca_loadings(panel, rf, path):
    pca = rf["pca"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    tn = C.CURVE_TENORS
    names = ["PC1 (уровень)", "PC2 (наклон)", "PC3 (кривизна)"]
    for i in range(C.N_PCA_RATE):
        ax[0].plot(tn, pca.components_[i], marker="o", ms=3, label=names[i])
    ax[0].set_title("Нагрузки главных компонент кривой")
    ax[0].set_xlabel("срок, лет"); ax[0].legend(fontsize=7)
    ev = pca.explained_variance_ratio_
    ax[1].bar(range(1, len(ev) + 1), ev * 100)
    ax[1].set_title("Доля объяснённой дисперсии, %")
    for i, v in enumerate(ev):
        ax[1].text(i + 1, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=8)
    ax[1].set_xlabel("компонента")
    fig.savefig(path); plt.close(fig)

def fig_factor_history(rf, path):
    F = rf["factors"]
    sel = ["RatePC1", "ST_SBER", "ST_GAZP", "FX_USD", "FX_EUR", "ST_LKOH"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for ax, c in zip(axes.ravel(), sel):
        ax.plot(F.index, F[c], lw=0.5)
        ax.set_title(c, fontsize=9)
    fig.suptitle("История инноваций риск-факторов (выборка)")
    fig.savefig(path); plt.close(fig)

def fig_corr_heatmap(rf, path):
    C_ = rf["factors"].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(C_, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C_))); ax.set_yticks(range(len(C_)))
    ax.set_xticklabels(C_.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(C_.columns, fontsize=7)
    for i in range(len(C_)):
        for j in range(len(C_)):
            ax.text(j, i, f"{C_.values[i,j]:.1f}", ha="center", va="center",
                    fontsize=5.5, color="black")
    fig.colorbar(im, fraction=0.046)
    ax.set_title("Корреляции риск-факторов")
    fig.savefig(path); plt.close(fig)

def fig_tails(rf, marg, path):
    sel = ["RatePC1", "ST_SBER", "FX_USD"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for j, c in enumerate(sel):
        x = rf["factors"][c].dropna().values
        # QQ против нормали
        stats.probplot(x, dist="norm", plot=axes[0, j])
        axes[0, j].set_title(f"QQ (Normal): {c}", fontsize=9)
        # гистограмма vs t vs normal
        axes[1, j].hist(x, bins=80, density=True, alpha=0.5)
        xs = np.linspace(x.min(), x.max(), 400)
        p = marg[c]["t"]; n = marg[c]["norm"]
        axes[1, j].plot(xs, stats.t.pdf(xs, p["df"], p["loc"], p["scale"]),
                        "r", lw=1.2, label=f"t (df={p['df']:.1f})")
        axes[1, j].plot(xs, stats.norm.pdf(xs, n["loc"], n["scale"]),
                        "g--", lw=1.0, label="Normal")
        axes[1, j].set_yscale("log"); axes[1, j].legend(fontsize=7)
        axes[1, j].set_title(f"Плотность (лог): {c}", fontsize=9)
    fig.suptitle("Тяжесть хвостов: эмпирика vs Стьюдент vs Нормаль")
    fig.savefig(path); plt.close(fig)


def fig_levels(panel, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    fx = panel["indices"]
    ax[0].plot(fx.index, fx["USDFIXME"], label="USD/RUB", lw=0.9)
    ax[0].plot(fx.index, fx["EURFIXME"], label="EUR/RUB", lw=0.9)
    ax[0].set_title("Курсы валют (тренд)"); ax[0].legend(fontsize=7)
    ax[1].plot(fx.index, panel["indices"]["IMOEX"], label="IMOEX", lw=0.9)
    ax[1].plot(fx.index, panel["indices"]["RTSI"], label="RTSI", lw=0.9)
    ax[1].set_title("Индексы (объясняющие)"); ax[1].legend(fontsize=7)
    fig.savefig(path); plt.close(fig)

def fig_backtest(bt, path):
    subs = ["total", "bonds", "stocks", "fx"]
    titles = {"total": "Весь портфель", "bonds": "Облигации",
              "stocks": "Акции", "fx": "Валюта"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, k in zip(axes.ravel(), subs):
        pnl = bt[f"PnL_{k}"] / 1e6
        var = -bt[f"VaR_{k}"] / 1e6
        brk = bt[f"breach_{k}"] == 1
        ax.plot(bt.index, pnl, lw=0.7, color="steelblue", label="P&L")
        ax.plot(bt.index, var, lw=0.9, color="darkred", label="-VaR 99%")
        ax.scatter(bt.index[brk], pnl[brk], color="red", s=24, zorder=5,
                   label="пробой")
        ax.set_title(f"{titles[k]} (пробоев: {int(brk.sum())})", fontsize=9)
        ax.legend(fontsize=7); ax.set_ylabel("млн руб.")
    fig.suptitle("Бэктест VaR 99% за 2025 г.: P&L против −VaR")
    fig.savefig(path); plt.close(fig)


def make_all(panel, rf, marg, bt=None, outdir=C.OUT_DIR):
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    fig_rates_history(panel, rf, os.path.join(outdir, "fig_rates_history.png")); paths["rates"] = 1
    fig_pca_loadings(panel, rf, os.path.join(outdir, "fig_pca_loadings.png"))
    fig_factor_history(rf, os.path.join(outdir, "fig_factor_history.png"))
    fig_corr_heatmap(rf, os.path.join(outdir, "fig_corr_heatmap.png"))
    fig_tails(rf, marg, os.path.join(outdir, "fig_tails.png"))
    fig_levels(panel, os.path.join(outdir, "fig_levels.png"))
    if bt is not None:
        fig_backtest(bt, os.path.join(outdir, "fig_backtest.png"))
    return outdir


if __name__ == "__main__":
    import data_loader as dl, risk_factors as rfm, dynamics as dyn
    panel = dl.build_panel()
    rf = rfm.build_risk_factors(panel)
    marg = dyn.fit_marginals(rf["factors"])
    make_all(panel, rf, marg)
    print("Графики сохранены в", C.OUT_DIR)
    print(os.listdir(C.OUT_DIR))