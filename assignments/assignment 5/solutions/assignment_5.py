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
spy = np.log(spy).diff() * 100  # Calculate log returns

# Plot the time series od returns
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(100 + spy.cumsum(), color=nupurple)
# ax.set_title("SPY Cumulative Log Returns")
ax.set_xlabel("Date")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("images/spy_log_returns.png", dpi=300, bbox_inches="tight")

# Question 1 -------------------------------------------------------------------
"""
In this section we will calculate the realized volatility (RV) of the SPY for multiple frequencies. As the frequency increases, the RV estimator becomes more precise. However, due to the presence of microstructure noise, the RV estimator can deviate from the true value if we increase the sampling frequency "too much".
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


# List of frequencies
freq_list = np.arange(1, 61)
# Calculate average RV over the sample for each frequency
avg_rv = np.array([calcRV(spy, str(x) + "min").mean() for x in freq_list])

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.regplot(
    x=freq_list,
    y=avg_rv,
    logx=True,
    ci=None,
    color=nupurple,
    line_kws={"linewidth": 1, "linestyle": "--", "color": "gray"},
    ax=ax,
)
ax.set_xlabel("Sampling Frequency (minutes)")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("images/volatility_signature.png", dpi=300, bbox_inches="tight")


# Question 2 -------------------------------------------------------------------
"""
We now change to study the intraday volatility "smile". We will use absolute returns as a measure of volatility and we calculate the average volatility for every minute of the trading day. 
"""


def calcIntradayVol(
    returns: pd.Series,
    freq: str = "1min",
    start_time=dt.time(10, 0),
    end_time=dt.time(16, 0),
):
    """
    Function to calculate intraday volatility over the market hours


    Args:
        returns (pd.Series): Series of log returns
        freq (str, optional): Frequency to aggregate returns. Defaults to '1min'.
        start_time (dt.time, optional): Start time of market hours. Defaults to dt.time(10, 0).
        end_time (dt.time, optional): End time of market hours. Defaults to dt.time(16, 0).

    Returns:
        pd.Series: Intraday volatility
    """
    # Upsample to minute frequency
    sample = returns.resample(freq).sum()
    # Average absolute retuns per minute
    vol = sample.abs().groupby(sample.index.time).sum() / 60
    # Subset for market hours
    vol = vol[(vol.index >= start_time) & (vol.index <= end_time)]
    return vol


def trend(y: np.array, x: np.array = None, order: int = 1):
    """
    Fit a polynomial trend to a time series.

    Args:
        y (np.array): Series to fit trend to
        x (np.array, optional): Optional series of indexes. Defaults to None.
        order (int, optional): Polynomial order. Defaults to 1.

    Returns:
        np.array: Fitted trend values
    """
    x = np.arange(len(y)) if x is None else x  # Use index if x is not provided
    coefs = np.polyfit(x, y, order)  # Fit polynomial
    return np.polyval(coefs, x)  # Fitted values


# Calculate intraday volatility for full sample
intraday_vol_full_sample = calcIntradayVol(spy)
time_index = pd.to_datetime(
    intraday_vol_full_sample.index.astype(str), format="%H:%M:%S"
)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    time_index[1:],
    intraday_vol_full_sample.values[1:],
    color=nupurple,
    label="First 2 Years",
)
ax.plot(
    time_index[1:],
    trend(intraday_vol_full_sample.values[1:], order=2),
    color="gray",
    linestyle="--",
    linewidth=2,
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("images/intraday_vol.png", dpi=300, bbox_inches="tight")

# Question 3 -------------------------------------------------------------------
""" 
We now check how the volatility smile have changed over the years. We calculate the average intraday volatility for the first two years of the sample and the last two years of the sample and plot them together.
"""

intraday_vol_first2y = calcIntradayVol(spy["1994":"1996"])  # First two years
intraday_vol_last2y = calcIntradayVol(spy["2021":"2023"])  # Last two years

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    time_index[1:],
    intraday_vol_first2y.values[1:],
    color=nupurple,
    label="1994-1995",
)
ax.plot(  # Add trendline
    time_index[1:],
    trend(intraday_vol_first2y.values[1:], order=2),
    color="gray",
    linestyle="--",
    linewidth=2,
)
ax.plot(
    time_index[1:],
    intraday_vol_last2y.values[1:],
    color="black",
    label="2021-2022",
)
ax.plot(  # Add trendline
    time_index[1:],
    trend(intraday_vol_last2y.values[1:], order=2),
    color="gray",
    linestyle="--",
    linewidth=2,
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.legend(ncol=3, frameon=False, loc="upper center")
# ax.set_title("Intraday Volatility for Sample Windows")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("images/intraday_vol_firstlast.png", dpi=300, bbox_inches="tight")

# Question 4 -------------------------------------------------------------------
"""
Finally we assess the persistence of the realized volatility measures. Using different measures of volatility, we plot their time series and calculate the ACF for up to 20 lags.
"""

# Calculate realized volatility
realized_volatility = calcRV(spy, "1min")

# Plot
fig, ax = plt.subplots(3, 1, figsize=(12, 12))
# Realized Variance
ax[0].plot(realized_volatility, color=nupurple, label="Volatility")
ax[0].plot(realized_volatility.rolling(30).mean(), color="gray", label="MA(30D)")
ax[0].set_title("Realized Variance")
ax[0].legend(frameon=False, ncol=2, loc="upper center")
# Realized Volatility
ax[1].plot(np.sqrt(realized_volatility), color=nupurple)
ax[1].plot(np.sqrt(realized_volatility).rolling(30).mean(), color="gray")
ax[1].set_title("Realized Volatility")
# Log Realized Volatility
ax[2].plot(np.log(realized_volatility), color=nupurple)
ax[2].plot(np.log(realized_volatility).rolling(30).mean(), color="gray")
ax[2].set_title("Realized Volatility (Log)")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("images/realized_volatility.png", dpi=300, bbox_inches="tight")

# Plot autocorrelogram

max_lags = 20  # Max number of lags to plot
fig, ax = plt.subplots(3, 1, figsize=(12, 12))
# Realized Variance
plot_acf(realized_volatility, ax=ax[0], lags=max_lags, color=nupurple)
ax[0].set_title("Realized Variance")
ax[0].set_ylim(-0.10, 1)
# Realized Volatility
plot_acf(np.sqrt(realized_volatility), ax=ax[1], lags=max_lags, color=nupurple)
ax[1].set_title("Realized Volatility")
ax[1].set_ylim(-0.10, 1)
# Log Realized Volatility
plot_acf(np.log(realized_volatility), ax=ax[2], lags=max_lags, color=nupurple)
ax[2].set_title("Realized Volatility (Log)")
ax[2].set_ylim(-0.10, 1)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("images/realized_volatility_acf.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(3, 1, figsize=(12, 12))
# Realized Variance
plot_pacf(realized_volatility, ax=ax[0], lags=max_lags, color=nupurple)
ax[0].set_title("Realized Variance")
ax[0].set_ylim(-0.10, 1)
# Realized Volatility
plot_pacf(np.sqrt(realized_volatility), ax=ax[1], lags=max_lags, color=nupurple)
ax[1].set_title("Realized Volatility")
ax[1].set_ylim(-0.10, 1)
# Log Realized Volatility
plot_pacf(np.log(realized_volatility), ax=ax[2], lags=max_lags, color=nupurple)
ax[2].set_title("Realized Volatility (Log)")
ax[2].set_ylim(-0.10, 1)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("images/realized_volatility_pacf.png", dpi=300, bbox_inches="tight")
