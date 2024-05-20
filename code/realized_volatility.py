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
from statsmodels.tsa.stattools import acf, pacf

sns.set_theme(context = "notebook",style = "whitegrid", palette = "colorblind", font_scale = 1.25, rc = {"figure.figsize": (12, 5)})

nupurple = "#492C7F"

# Data -------------------------------------------------------------------------

spy = pd.read_csv("../data/SPY_HF.zip", compression = 'zip', usecols = ['DATETIME', 'PRICE'])
spy.rename({x: x.lower() for x in spy.columns}, axis = 1, inplace = True)

spy['datetime'] = pd.to_datetime(spy['datetime'], format = '%d%b%Y:%H:%M:%S')
spy.set_index('datetime', inplace = True)
spy = spy.groupby(spy.index.date).apply(lambda x: x.resample('1min').ffill())
spy.reset_index(drop = True, level = 0, inplace = True)
spy.rename({'price': 'return'}, axis = 1, inplace = True)

spy = np.log(spy).diff()[1:]['return']  # Log price

def remove_outlier(x, cutoff = 0.002):
    from datetime import time
    for i,y in enumerate(x):
        if i > 0 and np.abs(y) > cutoff:
            x[i] = 0
    return x

spy = spy.groupby(spy.index.date).apply(remove_outlier)

# Plot the time series od returns
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(100 + spy.cumsum(), color=nupurple)
# ax.set_title("SPY Cumulative Log Returns")
ax.set_xlabel("Date")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("../images/spy_log_returns.png", dpi=300, bbox_inches="tight")


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
    real_var = (sample**2).resample("B").sum()
    real_var = real_var[real_var > 0].dropna()
    real_var *= 252  # Annualize realized volatility
    return real_var


# List of frequencies
freq_list = np.arange(1, 61)
# Calculate average RV over the sample for each frequency
avg_rv = np.array([calcRV(spy, str(x) + "min").mean() for x in freq_list])
coefs = np.polyfit(np.log(freq_list), avg_rv, 2)
trend = np.polyval(coefs, np.log(freq_list))

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(freq_list, avg_rv, marker = 'o', label = 'RV')
ax.plot(freq_list, trend, color = 'gray', linestyle = '--', linewidth = 2, label = 'Trend')
ax.set_xlabel("Sampling Frequency (minutes)")
ax.set_ylabel("Average RV")
ax.legend(loc = 'upper center', ncol = 2)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("../images/volatility_signature.png", dpi=300, bbox_inches="tight")


# Question 2 -------------------------------------------------------------------
"""
We now change to study the intraday volatility "smile". We will use absolute returns as a measure of volatility and we calculate the average volatility for every minute of the trading day. 
"""

def calcIntradayVol(
    returns: pd.Series,
    freq: int = 1,
    start_time=dt.time(9,35),
    end_time=dt.time(16,00),
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
    sample = returns.resample(str(freq) + 'min').sum()
    # Average absolute retuns per minute
    n = 60 / freq
    vol = sample.abs().groupby(sample.index.time).sum() / n
    # vol = (sample**2).groupby(sample.index.time).sum() / n
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
    time_index,
    intraday_vol_full_sample.values,
    color=nupurple,
)
ax.plot(
    time_index,
    trend(intraday_vol_full_sample.values, order=2),
    color="gray",
    linestyle="--",
    linewidth=2,
)
ax.legend(loc = 'upper center', ncol = 2, frameon = False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("../images/intraday_vol.png", dpi=300, bbox_inches="tight")

# Question 3 -------------------------------------------------------------------
""" 
We now check how the volatility smile have changed over the years. We calculate the average intraday volatility for the first two years of the sample and the last two years of the sample and plot them together.
"""

intraday_vol_years = spy.resample('2Y').apply(calcIntradayVol).unstack().T

# Plot
fig, ax = plt.subplots(figsize = (12,5))
fig.tight_layout()
ax.plot(
    time_index,
    intraday_vol_years.iloc[:,0],
    color="C0",
    label="First Two Years",
)
ax.plot(  # Add trendline
    time_index,
    trend(intraday_vol_years.iloc[:,0], order=2),
    color="gray",
    linestyle="--",
    linewidth=2,
)
ax.plot(
    time_index,
    intraday_vol_years.iloc[:,-1],
    label="Last Two Years",
    color = "C1"
)
ax.plot(  # Add trendline
    time_index,
    trend(intraday_vol_years.iloc[:,-1], order=2),
    color="gray",
    linestyle="--",
    linewidth=2,
)
ax.legend(loc = 'upper center', ncol = 2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# ax.legend(ncol=3, frameon=False, loc="upper center")
# ax.set_title("Intraday Volatility for Sample Windows")
# fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust margins
plt.savefig("../images/intraday_vol_firstlast.png", dpi=300, bbox_inches="tight")

# Question 4 -------------------------------------------------------------------
"""
Finally we assess the persistence of the realized volatility measures. Using different measures of volatility, we plot their time series and calculate the ACF for up to 20 lags.
"""

# Calculate realized volatility
realized_volatility = calcRV(spy, "3min")

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
plt.savefig("../images/realized_volatility.png", dpi=300, bbox_inches="tight")

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
plt.savefig("../images/realized_volatility_acf.png", dpi=300, bbox_inches="tight")

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
plt.savefig("../images/realized_volatility_pacf.png", dpi=300, bbox_inches="tight")
