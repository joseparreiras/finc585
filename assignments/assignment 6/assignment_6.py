"""
FINC 585-3: Asset Pricing
Assignment 5 (Solutions)
Prof. Torben Andersen
TA: Jose Antunes-Neto
"""

# Modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import datetime as dt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

nupurple = "#492C7F"

# Data -------------------------------------------------------------------------

# Sample period
start = "1993-02-01"
end = "2023-01-31"

spy = pd.read_csv(
    "data/SPY_HF.csv", dtype={"DATE": "str", "TIME": "str", "PRICE": float}
)  # Read the data
# Combine DATE and TIME columns for index
spy.index = pd.to_datetime(spy["DATE"] + spy["TIME"], format="%Y%m%d%H%M")
# Drop unnecessary columns
spy = spy["PRICE"]
spy = spy[start:end]  # Subset for sample period
spy = np.log(spy)  # Log price

# 1) ---------------------------------------------------------------------------
"""
In this question we just need to calculate the mean realized volatility for the same sample that we had in the previous assignment. 
"""


def calcRV(returns: pd.Series, sampling_freq: str = "1min"):
    """
    Calculate the time series of realized volatilities over a frequency.

    Args:
        returns (pd.Series): Series of log returns
        freq (str, optional): Frequency to calculate RV. Defaults to '1min'.

    Retruns:
        (pd.Series): Series of realized volatilities (RV)
    """
    # Upsample to desired frequency
    sample = returns.resample(sampling_freq).sum()
    # Use pd.Series.sum to set sums of empty series to NA
    # Sum squared returns daily
    real_var = (sample**2).resample("1D").sum()
    real_var = real_var[real_var > 0]  # Drop NA
    real_var *= 252  # Annualize realized volatility
    return real_var


# Calculate realized variance for 5-min sampling
trading_rv = calcRV(spy.diff() * 100, "5min")
trading_rv.rename("RV", inplace=True)  # Rename series

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(spy.resample("5min").last().diff() * 100 * 252, color=nupurple)
ax.yaxis.set_major_formatter("{:.0f}%".format)
plt.savefig("images/spy_returns.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(trading_rv, color=nupurple)
ax.yaxis.set_major_formatter("{:.0f}%".format)
plt.savefig("images/trading_rv.png", dpi=300, bbox_inches="tight")

# 2) ---------------------------------------------------------------------------
"""
We now add the overnight realized volatility to the trading hours volatility to obtain the total realized volatility. We will use the opening price of the day as the overnight return and subtract the closing price of the previous day. We will then square the overnight return to obtain the overnight realized variance.
"""
# Aggregate S&P data to daily frequency
spy_daily = spy.resample("1D").agg({"close": "last", "open": "first"})
spy_daily = spy_daily.dropna()  # Drop NA
# Calculate overnight returns
overnight_return = (spy_daily["open"] - spy_daily["close"].shift(1)) * 100
# Transform to overnight volatility
overnight_rv = overnight_return**2
overnight_rv *= 252  # Annualize
overnight_rv = overnight_rv[trading_rv.index]  # Overlap with realized vol
overnight_rv.rename("Overnight RV", inplace=True)  # Rename series
# Calculate total realized volatility by summing both
total_rv = trading_rv + overnight_rv
total_rv.rename("Total RV", inplace=True)  # Rename series

# Create a table with the summary statistics for the measures of volatility
rv_stats = pd.concat([trading_rv[1:], overnight_rv, total_rv], axis=1).describe()
rv_stats.to_latex("tables/rv_stats.tex", float_format="{:.2f}%".format)  # Save to latex

# 3) ---------------------------------------------------------------------------
"""
This question asks us to compare the realized volatility series with the VIX index to get a sense of the volatility premium. The variance swap contract goes long in the RV and pays VIX at the maturity, so the variance risk premium corresponds to the opposite return of this strategy. 
It is important to note that the VIX refers to the expected volatility of the market for the next 30 days. To compare it to the realized volatility measure, we first make a moving average of the RV in the last 30 days and lag the series of the VIX to match the maturities.
We download the VIX from the CBOE website and obtain its closing price of the day. 
"""

# Download data
url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
vix = pd.read_csv(url, index_col=0, parse_dates=True)  # Read data
vix = vix["CLOSE"]  # Keep only close price
vix.rename("VIX", inplace=True)  # Rename series
vix = vix[start:end]  # Overlap with realized vol

# Calculate volatility premium (VIX needs to be lagged)
vol_premium = vix.shift(30) - np.sqrt(total_rv).rolling(30).mean()
vol_premium.rename("Volatility Risk Premium", inplace=True)  # Rename series

# Calculate variance premium
var_premium = (vix**2).shift(30) - total_rv.rolling(30).mean()
var_premium.rename("Variance Risk Premium", inplace=True)  # Rename series

# Export descriptive statistics to latex tables
vol_premium_stats = pd.concat([np.sqrt(total_rv), vix, vol_premium], axis=1).describe()
vol_premium_stats.to_latex(
    "tables/vol_premium_stats.tex", float_format="{:.2f}%".format
)

var_premium_stats = pd.concat([total_rv, vix**2, var_premium], axis=1).describe()
var_premium_stats.to_latex(
    "tables/var_premium_stats.tex", float_format="{:.2f}%".format
)

# 4) ---------------------------------------------------------------------------
"""
We finish by plotting the graphs of both series and comparing their realizations.
"""

# Plot realized volatility
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.sqrt(total_rv).rolling(30).mean(), label=r"$RV_t$", color=nupurple)
ax.plot(vix.shift(30), label=r"$VIX_{t-30}$", color="orange")
ax.legend(ncol=2, frameon=False, loc="upper center")
ax.set_xlabel("Date")
ax.yaxis.set_major_formatter("{:.0f}%".format)
plt.savefig("images/rv_and_vix.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(vol_premium, color=nupurple)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Date")
ax.yaxis.set_major_formatter("{:.0f}%".format)
plt.savefig("images/vol_premium.png", dpi=300, bbox_inches="tight")
