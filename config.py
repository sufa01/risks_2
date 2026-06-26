# -*- coding: utf-8 -*-
"""
Единая конфигурация проекта
"""
from __future__ import annotations
import os
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SRC_DIR, "data")
OUT_DIR = os.path.join(SRC_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "rates":   os.path.join(DATA_DIR, "Процентные ставки.xlsx"),
    "bonds":   os.path.join(DATA_DIR, "Котировки облигации.csv"),
    "stocks":  os.path.join(DATA_DIR, "Котировки акции.csv"),
    "indices": os.path.join(DATA_DIR, "Индексы и курсы.csv"),
    "brent":   os.path.join(DATA_DIR, "нефть-brent.xlsx"),
    "options": os.path.join(DATA_DIR, "MIX_Options_2025-10-17.csv"),
    "bond_isin": os.path.join(DATA_DIR, "Календарь облигационных выплат.csv"),
}

# Воспроизводимость
SEED = 42
def get_rng(seed: int | None = None) -> np.random.Generator:
    """Единый генератор случайных чисел проекта."""
    return np.random.default_rng(SEED if seed is None else seed)

# Состав портфеля
# 5 ОФЗ по 10 млн руб., 10 акций по 1 млн руб.,
# 100 млн руб. в долларах и 100 млн руб. в евро
BOND_SECIDS = [
    "SU26218RMFS6", "SU26224RMFS4", "SU26230RMFS1",
    "SU26231RMFS9", "SU26233RMFS5",
]
STOCK_TICKERS = [
    "GAZP", "GMKN", "LKOH", "MGNT", "NVTK",
    "PLZL", "ROSN", "SBER", "SNGS", "TATN",
]
FX_SECIDS = ["USDFIXME", "EURFIXME"]

NOTIONAL_PER_BOND = 10_000_000.0 
NOTIONAL_PER_STOCK = 1_000_000.0  
NOTIONAL_PER_FX = 100_000_000.0  

# Целевые веса 
def target_notionals() -> dict:
    d = {}
    for s in BOND_SECIDS:
        d[("bond", s)] = NOTIONAL_PER_BOND
    for t in STOCK_TICKERS:
        d[("stock", t)] = NOTIONAL_PER_STOCK
    for f in FX_SECIDS:
        d[("fx", f)] = NOTIONAL_PER_FX
    return d

TOTAL_NOTIONAL = (
    len(BOND_SECIDS) * NOTIONAL_PER_BOND
    + len(STOCK_TICKERS) * NOTIONAL_PER_STOCK
    + len(FX_SECIDS) * NOTIONAL_PER_FX
)

# Параметры оценки риска
RISK_DATE = "2025-12-02"        # дата оценки риска
HORIZONS = [1, 10]              # горизонты в торговых днях
VAR_LEVEL = 0.99               # уровень VaR
ES_LEVEL = 0.975               # уровень Expected Shortfall
N_SIM = 50_000                 # число траекторий Монте-Карло
N_SIM_BACKTEST = 20_000        # число траекторий в бэктесте

BACKTEST_YEAR = 2025           # год для бэктеста
ROLL_WINDOW = 500              # окно оценки параметров моделей при бэктесте
REFIT_EVERY = 5                # как часто переоценивать маргиналы в бэктесте 

TRADING_DAYS_PER_YEAR = 252

# Сроки узлов кривой бескупонной доходности
CURVE_TENORS = [0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10, 15, 20, 30]

# Число главных компонент кривой ставок
N_PCA_RATE = 3

# Нижний порог числа степеней свободы t-распределения при MLE
T_DF_FLOOR = 3.0