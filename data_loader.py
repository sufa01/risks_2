# -*- coding: utf-8 -*-
"""
1) загрузка и очистка собранных данных.
  * Процентные ставки.xlsx          — кривая бескупонной доходности (КБД) MOEX,
                                       сроки 0.25..30 лет, % годовых.  Первоисточник:
                                       расчёт MOEX по методике Nelson-Siegel-Svensson
                                       на основе сделок с ОФЗ
  * Котировки облигации.csv         — дневные котировки 5 ОФЗ (TQOB), MOEX ISS.
  * Котировки акции.csv             — дневные котировки 10 акций (TQBR), MOEX ISS.
  * Индексы и курсы.csv             — IMOEX, RTSI, курсы USD/EUR (фиксинг MOEX), MOEX ISS.
  * нефть-brent.xlsx                — цена Brent
  * MIX_Options_2025-10-17.csv      — фьючерс и опционы на индекс MIX (FORTS) на 1 день
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config as C

# Кривая бескупонной доходности
def load_rate_curve() -> pd.DataFrame:

    raw = pd.read_excel(C.FILES["rates"], header=None)
    tenors = raw.iloc[1, :].dropna().astype(float).tolist()
    n = len(tenors)
    body = raw.iloc[2:, : n + 1].copy()
    body.columns = ["date"] + tenors
    body["date"] = pd.to_datetime(body["date"])
    body = body.set_index("date").sort_index()
    body = body.astype(float) / 100.0 
    body.columns = [float(t) for t in tenors]
    return body

# Облигации
def _bond_clean_price(df: pd.DataFrame) -> pd.Series:
    price = df["CLOSE"].copy()
    for col in ["LEGALCLOSEPRICE", "WAPRICE", "MARKETPRICE3", "MARKETPRICE2"]:
        price = price.fillna(df[col]) if col in df.columns else price
    return price


def load_bonds() -> dict:
    """
    Возвращает словарь:
      'clean'   DataFrame чистых цен (% номинала), columns = SECID
      'accint'  DataFrame накопленного купонного дохода (руб.)
      'meta'    DataFrame со статикой по каждой облигации (номинал, купон, погашение)
    """
    df = pd.read_csv(C.FILES["bonds"], parse_dates=["TRADEDATE", "MATDATE"])
    df = df[df["SECID"].isin(C.BOND_SECIDS)].copy()
    clean, accint = {}, {}
    meta_rows = []
    for sid, g in df.groupby("SECID"):
        g = (g.sort_values("TRADEDATE")
               .drop_duplicates("TRADEDATE", keep="last")
               .set_index("TRADEDATE"))
        clean[sid] = _bond_clean_price(g)
        accint[sid] = g["ACCINT"]
        # величина купона: берём COUPONVALUE (совпадает с приростом ACCINT)
        coupon_value = float(g["COUPONVALUE"].dropna().iloc[-1])
        face = float(g["FACEVALUE"].dropna().iloc[-1])
        meta_rows.append({
            "SECID": sid,
            "SHORTNAME": g["SHORTNAME"].iloc[0],
            "ISIN": g["ISIN"].dropna().iloc[0] if g["ISIN"].notna().any() else None,
            "MATDATE": g["MATDATE"].dropna().iloc[-1],
            "FACEVALUE": face,
            "COUPONVALUE": coupon_value,      # руб. за купонный период (полугодие)
            "COUPONPERCENT": float(g["COUPONPERCENT"].dropna().iloc[-1]),
            "FREQ": 2,                         # ОФЗ-ПД: полугодовой купон
        })
    clean = pd.DataFrame(clean).sort_index()
    accint = pd.DataFrame(accint).sort_index()
    meta = pd.DataFrame(meta_rows).set_index("SECID").loc[C.BOND_SECIDS]
    return {"clean": clean, "accint": accint, "meta": meta}


def build_coupon_schedule(meta_row: pd.Series, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Восстанавливает расписание будущих выплат облигации, идя от даты погашения
    """
    mat = pd.Timestamp(meta_row["MATDATE"])
    face = meta_row["FACEVALUE"]
    coupon = meta_row["COUPONVALUE"]
    step = pd.DateOffset(months=6)
    dates = []
    d = mat
    while d > asof:
        dates.append(d)
        d = d - step
    dates = sorted(dates)
    cf = [coupon] * len(dates)
    if dates:
        cf[-1] += face
    return pd.DataFrame({"date": dates, "cf": cf})

# Акции
def _adjust_splits(prices: pd.DataFrame, lo: float = 0.2, hi: float = 5.0) -> pd.DataFrame:
    """
    Корректировка дроблений акций (сплитов).  Если цена за день меняется
    более чем в 5 раз (ratio<0.2 или >5) — это корпоративное действие, а не
    рыночное движение.
    Обвал рынка 24.02.2022 (ratio~0.63) под порог НЕ попадает и
    остаётся как реальная доходность.
    """
    adj = prices.copy()
    for c in adj.columns:
        s = adj[c]
        ratio = s / s.shift(1)
        events = ratio[(ratio < lo) | (ratio > hi)].dropna()
        for d, r in events.items():
            adj.loc[adj.index < d, c] = adj.loc[adj.index < d, c] * r
    return adj


def load_stocks() -> pd.DataFrame:
    df = pd.read_csv(C.FILES["stocks"], parse_dates=["TRADEDATE"])
    df = df[df["TICKER"].isin(C.STOCK_TICKERS)]
    price = df["CLOSE"].fillna(df["LEGALCLOSEPRICE"]).fillna(df["WAPRICE"])
    out = (df.assign(price=price)
             .pivot_table(index="TRADEDATE", columns="TICKER", values="price")
             .sort_index())
    out = out[C.STOCK_TICKERS]
    return _adjust_splits(out)

# Индексы и курсы валют
def load_indices() -> pd.DataFrame:
    df = pd.read_csv(C.FILES["indices"], parse_dates=["TRADEDATE"])
    out = (df.pivot_table(index="TRADEDATE", columns="SECID", values="CLOSE")
             .sort_index())
    keep = [c for c in ["IMOEX", "RTSI", "USDFIXME", "EURFIXME"] if c in out.columns]
    return out[keep]

def load_brent() -> pd.Series:
    df = pd.read_excel(C.FILES["brent"])
    df.columns = ["date", "brent"]
    s = (df.assign(date=pd.to_datetime(df["date"]))
           .set_index("date")["brent"].sort_index())
    return s

# Фьючерс и опционы (для бонусного задания)
def load_options() -> dict:
    df = pd.read_csv(C.FILES["options"])
    date = pd.Timestamp(df["TRADEDATE"].iloc[0])
    fut = df[df["MARKET_TYPE"] == "forts"].iloc[0]
    future = {
        "secid": fut["SECID"],
        "close": float(fut["CLOSE"]),
        "settle": float(fut["SETTLEPRICE"]),
        "shortname": fut["SHORTNAME"],
    }
    opt = df[df["MARKET_TYPE"] == "options"].copy()

    def parse(secid: str):
        body = secid[2:]                      
        i = 0
        while i < len(body) and body[i].isdigit():
            i += 1
        strike = int(body[:i])
        letter = body[-2]                     
        is_call = letter in "ABCDEFGHIJKL"     
        return strike, ("C" if is_call else "P")

    strikes, types = [], []
    for sid in opt["SECID"]:
        k, t = parse(sid)
        strikes.append(k); types.append(t)
    opt["STRIKE"] = strikes
    opt["TYPE"] = types
    opt["PRICE"] = opt["SETTLEPRICE"].fillna(opt["CLOSE"])
    return {"date": date,
            "future": future,
            "options": opt[["SECID", "STRIKE", "TYPE", "PRICE", "CLOSE",
                            "SETTLEPRICE", "OPENPOSITION", "VOLUME"]].reset_index(drop=True)}

# Сборка единой панели
def build_panel() -> dict:
    rates = load_rate_curve()
    bonds = load_bonds()
    stocks = load_stocks()
    idx = load_indices()
    brent = load_brent()
    common = stocks.index
    for obj in [bonds["clean"], idx]:
        common = common.intersection(obj.index)
    common = common.sort_values()

    def align(x):
        return x.reindex(common).ffill()

    panel = {
        "dates": common,
        "rates": rates.reindex(common).ffill().bfill(),
        "bond_clean": align(bonds["clean"]),
        "bond_accint": align(bonds["accint"]),
        "bond_meta": bonds["meta"],
        "stocks": align(stocks),
        "indices": align(idx),
        "brent": brent.reindex(common).ffill(),
    }
    return panel


if __name__ == "__main__":
    p = build_panel()
    print("Общий календарь:", p["dates"].min().date(), "->", p["dates"].max().date(),
          "| дней:", len(p["dates"]))
    print("Облигации:", list(p["bond_clean"].columns))
    print("Акции:", list(p["stocks"].columns))
    print("Кривая, сроки:", list(p["rates"].columns))
    print(p["bond_meta"][["SHORTNAME", "MATDATE", "COUPONVALUE", "FACEVALUE"]])