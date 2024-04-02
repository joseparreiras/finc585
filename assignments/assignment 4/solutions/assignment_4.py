"""
FINC 585-3: Asset Pricing
Assignment 4 (Solutions)
Prof. Torben Andersen
TA: Jose Antunes-Neto
"""

# Modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

nupurple = "#492C7F"

# Simulation -------------------------------------------------------------------
"""
In this question we will exemplify the consistency of the Fama-Macbeth estimator using a simulated environment.
"""

# Set the parameters
nobs = 20  # N = 20
tobs = 5000  # T = 5000
beta = 2  # True beta

# Generate the errors
np.random.seed(int("02011921"))  # Set seed
mu = np.random.standard_normal((nobs, 1))  # X fixed effect
eta = np.random.standard_normal((nobs, tobs))  # X error
gamma = np.random.standard_normal((nobs, 1))  # Epsilon fixed effect
nu = np.random.standard_normal((nobs, tobs))  # Epsilon error

# Create a function to estimate the Fama-Macbeth betas


def estimate_fm(y: np.array, x: np.array, method: str = "traditional"):
    """
    Estimates the Fama-Macbeth betas

    Args:
        y (np.array): Dependent variable
        x (np.array): Independent variable
        method (str, optional): Method. Defaults to "traditional". Alternative to "demeaned".

    Returns:
        float: Estimated beta
    """
    from statsmodels.api import OLS, add_constant

    tobs = x.shape[1]  # Number of observations
    beta_fm = 0  # Initialize beta
    if method == "demeaned":
        x = x - np.mean(x, axis=1).reshape(-1, 1)  # Demeaned x
        y = y - np.mean(y, axis=1).reshape(-1, 1)  # Demeaned y
    for t in np.arange(tobs):
        beta_t = np.corrcoef(x[:, t], y[:, t])[0, 1] * np.std(y[:, t]) / np.std(x[:, t])
        beta_fm += beta_t / tobs  # Average beta_t
    return beta_fm


# (a)
# Compute (x,epsilon) according to equation 2
x = mu + eta
e = gamma + nu
y = beta * x + e

# Estimate the FM betas
tobs_range = np.arange(100, tobs + 1, 100)
beta_fm = np.array(
    [estimate_fm(y[:, :t], x[:, :t], method="traditional") for t in tobs_range]
)
beta_dfm = np.array(
    [estimate_fm(y[:, :t], x[:, :t], method="demeaned") for t in tobs_range]
)

# Plot
# Estimates
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(beta, color="black", linestyle="--")
ax.plot(tobs_range, beta_fm, color=nupurple, label="Traditional")
ax.plot(tobs_range, beta_dfm, linestyle="--", color=nupurple, label="Demeaned")
ax.legend(loc="upper center", frameon=False, ncol=2)
ax.set_ylim(1.92, 2.04)
ax.set_xlabel("Number of Observations")
plt.savefig("images/beta_fm_a.png", dpi=1200, bbox_inches="tight")

# (b)
"""
We now remove the firm fixed effect from x and repeat the exercise
"""
x = eta
y = beta * x + e

tobs_range = np.arange(100, tobs + 1, 100)
beta_fm = np.array(
    [estimate_fm(y[:, :t], x[:, :t], method="traditional") for t in tobs_range]
)
beta_dfm = np.array(
    [estimate_fm(y[:, :t], x[:, :t], method="demeaned") for t in tobs_range]
)

# Plot
# Estimates
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(beta, color="black", linestyle="--")
ax.plot(tobs_range, beta_fm, color=nupurple, label="Traditional")
ax.plot(tobs_range, beta_dfm, linestyle="--", color=nupurple, label="Demeaned")
ax.legend(loc="upper center", frameon=False, ncol=2)
ax.set_ylim(1.92, 2.04)
ax.set_xlabel("Number of Observations")
plt.savefig("images/beta_fm_b.png", dpi=1200, bbox_inches="tight")

# (c)
"""
I now remove the fixed effect from the error term.
"""
x = mu + eta
e = nu
y = beta * x + e

tobs_range = np.arange(100, tobs + 1, 100)
beta_fm = np.array(
    [estimate_fm(y[:, :t], x[:, :t], method="traditional") for t in tobs_range]
)
beta_dfm = np.array(
    [estimate_fm(y[:, :t], x[:, :t], method="demeaned") for t in tobs_range]
)

# Plot
# Estimates
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(beta, color="black", linestyle="--")
ax.plot(tobs_range, beta_fm, color=nupurple, label="Traditional")
ax.plot(tobs_range, beta_dfm, linestyle="--", color=nupurple, label="Demeaned")
ax.legend(loc="upper center", frameon=False, ncol=2)
ax.set_ylim(1.92, 2.04)
ax.set_xlabel("Number of Observations")
plt.savefig("images/beta_fm_c.png", dpi=1200, bbox_inches="tight")
