import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator
from tabulate import tabulate


TICKER = "SPY"               # "^NSEI" for Nifty
START, END = "2015-01-01", "2023-12-31"
SHORT, LONG = 20, 50
COST_BPS = 0.001             # 10 bps round-trip


price = yf.download(TICKER, start=START, end=END, auto_adjust=False)["Adj Close"].squeeze()

def run(name, fast, slow):
    df = pd.DataFrame(index=price.index)
    df["p"] = price

    if name == "SMA":
        df["fast"] = price.rolling(fast).mean()
        df["slow"] = price.rolling(slow).mean()
    else:
        df["fast"] = EMAIndicator(price, window=fast).ema_indicator()
        df["slow"] = EMAIndicator(price, window=slow).ema_indicator()

    df["pos"] = np.where(df["fast"] > df["slow"], 1, 0).astype(float)

    ret = df["p"].pct_change()
    strat = df["pos"].shift(1) * ret - df["pos"].diff().abs() * COST_BPS
    curve = (1 + strat).cumprod()

    def ann(r): return (1 + r).prod() ** (252 / r.count()) - 1
    cagr   = ann(strat.dropna())
    sharpe = cagr / (strat.std() * np.sqrt(252))
    max_dd = (curve / curve.cummax() - 1).min()
    trades = int((df["pos"].diff() != 0).sum() / 2)

    return dict(Name=name, CAGR=cagr, Sharpe=sharpe, MaxDD=max_dd, Trades=trades, Curve=curve)

sma = run("SMA", SHORT, LONG)
ema = run("EMA", SHORT, LONG)
bh  = (1 + price.pct_change()).cumprod()

# table
print("\n===== SMA vs EMA =====")
print(tabulate([{"Name": sma["Name"],
                 "CAGR": f"{sma['CAGR']:.2%}",
                 "Sharpe": f"{sma['Sharpe']:.2f}",
                 "MaxDD": f"{sma['MaxDD']:.2%}",
                 "Trades": sma["Trades"]},
                {"Name": ema["Name"],
                 "CAGR": f"{ema['CAGR']:.2%}",
                 "Sharpe": f"{ema['Sharpe']:.2f}",
                 "MaxDD": f"{ema['MaxDD']:.2%}",
                 "Trades": ema["Trades"]}],
               headers="keys", tablefmt="plain"))
# plot
plt.style.use("default")
plt.figure(figsize=(10,5))
plt.plot(bh, label="Buy & Hold", color="grey")
plt.plot(sma["Curve"], label=f"SMA-{SHORT}/{LONG}")
plt.plot(ema["Curve"], label=f"EMA-{SHORT}/{LONG}")
plt.yscale("log")
plt.legend(); plt.tight_layout()
plt.savefig("curves.png", dpi=300)
plt.show()