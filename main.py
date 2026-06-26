# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import config as C
import data_loader as dl
import risk_factors as rfm
import dynamics as dyn
import pricing as pr
import risk_measures as rm
import backtesting as bt
import bonus as bn
import plots

def run_all(fast: bool = False) -> dict:
    res = {}
    rng = C.get_rng()

    print("1. Загрузка и очистка данных")
    panel = dl.build_panel()
    res["panel"] = panel
    print(f"Торговых дней: {len(panel['dates'])}  "
          f"({panel['dates'].min().date()} — {panel['dates'].max().date()})")
    print(f"Облигаций: {len(C.BOND_SECIDS)}, акций: {len(C.STOCK_TICKERS)}, "
          f"валют: {len(C.FX_SECIDS)}; целевой объём портфеля: "
          f"{C.TOTAL_NOTIONAL/1e6:.0f} млн руб.")

    # 2: риск-факторы, PCA, описательная статистика
    print("2  Риск-факторы, PCA, описательная статистика")
    rf = rfm.build_risk_factors(panel)
    res["rf"] = rf
    ev = rf["pca"].explained_variance_ratio_
    print(f"PCA кривой: 12 сроков -> {C.N_PCA_RATE} компоненты, "
          f"объяснено {ev.sum()*100:.2f}% дисперсии "
          f"(уровень {ev[0]*100:.1f}%, наклон {ev[1]*100:.1f}%, кривизна {ev[2]*100:.1f}%)")
    ds = rfm.descriptive_stats(rf)
    tails = rfm.tail_heaviness(rf)
    corr = rfm.correlation_matrix(rf)
    seas = rfm.seasonality_trend(rf)
    ds.to_csv(C.OUT_DIR + "/p2_descriptive_stats.csv")
    tails.to_csv(C.OUT_DIR + "/p2_tail_heaviness.csv")
    corr.to_csv(C.OUT_DIR + "/p2_correlation.csv")
    seas.to_csv(C.OUT_DIR + "/p2_seasonality_trend.csv")
    print(f"Всего риск-факторов: {rf['factors'].shape[1]} "
          f"({rf['factors'].shape[0]} набл.)")
    print(f"Стационарность (ADF p<0.05): {(ds['ADF_p']<0.05).sum()}/{len(ds)} факторов")
    print(f"Не-нормальность (JB p<0.05): {(ds['JB_p']<0.05).sum()}/{len(ds)} факторов")
    res.update(descriptive=ds, tails=tails, corr=corr)

    # 3: модели динамики + MLE 
    print("3  Стохастические модели динамики (MLE)")
    marg = dyn.fit_marginals(rf["factors"])
    mtab = dyn.marginals_table(marg)
    R = dyn.gaussian_copula_corr(rf["factors"], marg)
    mtab.to_csv(C.OUT_DIR + "/p3_marginals.csv")
    pd.DataFrame(R, index=rf["factors"].columns,
                 columns=rf["factors"].columns).to_csv(C.OUT_DIR + "/p3_copula_corr.csv")
    res.update(marg=marg, mtab=mtab, R=R)
    print(f"Маргиналы: распределение Стьюдента (MLE) лучше нормали по AIC "
          f"для {int(mtab['t_better'].sum())}/{len(mtab)} факторов")
    print(f"Диапазон степеней свободы df: "
          f"{mtab['t_df'].min():.1f}–{mtab['t_df'].max():.1f} (тяжёлые хвосты)")

    # 4: ценообразование и точность
    print("4  Справедливая стоимость инструментов и точность")
    asof = pd.Timestamp(C.RISK_DATE)
    acc = pr.bond_pricing_accuracy(panel)
    acc.to_csv(C.OUT_DIR + "/p4_bond_accuracy.csv", index=False)
    spreads = pr.calibrate_bond_spreads(panel, asof)
    print(f"Облигации по КБД: средняя |ошибка| = "
          f"{acc['err_pct_face'].abs().mean():.2f}% номинала; после калибровки "
          f"z-спреда модель = рынок (спреды {spreads.min()*100:.2f}…{spreads.max()*100:.2f}%)")
    print("Акции — цена = риск-фактор (полная переоценка); "
          "валюта — стоимость = объём×курс.")
    res.update(bond_accuracy=acc, spreads=spreads)

    # 5: VaR / ES на 02.12.2025
    print("5  VaR (99%) и ES (97.5%) на 02.12.2025, горизонты 1 и 10 дней")
    rep = rm.build_repricer(panel, rf, asof)
    risk_tables = {}
    for h in C.HORIZONS:
        rr = rm.risk_report(rep, rf["factors"], marg, R, h, C.N_SIM, rng)
        rr.to_csv(C.OUT_DIR + f"/p5_risk_{h}d.csv")
        risk_tables[h] = rr
        tot = rr.loc["total"]
        print(f"\nГоризонт {h} дн.:")
        print(rr[["VaR_99", "ES_975"]].round(0).to_string())
        print(f"  VaR={tot['VaR_99']/1e6:.2f} млн ({tot['VaR_99']/C.TOTAL_NOTIONAL*100:.2f}%), "
              f"ES={tot['ES_975']/1e6:.2f} млн ({tot['ES_975']/C.TOTAL_NOTIONAL*100:.2f}%)")
    res["risk_tables"] = risk_tables

    # 6–7: бэктест и тесты 
    print("6–7  Бэктест VaR за 2025 г. и статистические тесты")
    btab = bt.run_backtest(panel, verbose=True)
    btab.to_csv(C.OUT_DIR + "/p6_backtest_2025.csv")
    summ = bt.breach_summary(btab)
    tests = bt.run_all_tests(btab)
    summ.to_csv(C.OUT_DIR + "/p6_breach_summary.csv")
    tests.to_csv(C.OUT_DIR + "/p7_tests.csv")
    print("\nСводка пробоев:")
    print(summ[["n_days", "breaches", "expected_breaches"]].round(2).to_string())
    print("\nТесты (p>0.05 => оценка корректна):")
    print(tests[["breaches", "Kupiec_p", "Christ_cc_p", "DQ_p"]].to_string())
    res.update(backtest=btab, breach_summary=summ, tests=tests)

    # 8: бонус
    print("8  Бонус: опционы (Блэк-76) и облигации со встроенными опционами")
    opt1 = bn.price_option_portfolio(panel)
    emb = bn.price_embedded_option_bonds(panel)
    res.update(bonus_options=opt1, bonus_embedded=emb)
    print(f"Опционы на фьючерс MIX: F={opt1['F']:.0f}, T={opt1['T']:.3f}, "
          f"калибр. σ={opt1['sigma']:.3f}")
    print(opt1["observed"][["SECID", "TYPE", "STRIKE", "PRICE",
                            "impl_vol", "model_price"]].round(2).to_string(index=False))
    print(f"\nОблигация {emb['sid']}: обычная={emb['plain']:.2f}, "
          f"putable={emb['putable']:.2f} (+{emb['putable_premium']:.2f}), "
          f"callable={emb['callable']:.2f} (-{emb['callable_discount']:.2f}) руб.")

    # Графики 
    plots.make_all(panel, rf, marg, bt=btab)
    print("Рисунки PNG сохранены в", C.OUT_DIR)
    return res

if __name__ == "__main__":
    run_all()