"""
FINC 585-3: Asset Pricing
Assignment 3 (Solutions)
Prof. Torben Andersen
TA: Jose Antunes-Neto
"""

# Modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Problem 2. -------------------------------------------------------------------
"""
In this question I perform a Monte Carlo study to exemplify the bias of the OLS estimator in the presence of persistence in the regressors.
The model takes the form of
y(t) = alpha + beta*x(t-1) + u(t)
x(t) = rho*x(t-1) + v(t)
I take alpha = 0 and beta = 1 at all times.
"""

np.random.seed(int("02011921"))

beta = 1  # OLS coefficient
sigma_u = 1.0  # Std. dev. of the error u
sigma_v = 1.0  # Std. dev. of the error v


def simulate_model(nobs: int, sigma_uv: float, rho: float, init=(0, 0)):
    """
    Function to simulate the model

    Args:
            nobs(int): Number of observations(T)
            sigma_uv(float): Covariance between errors u and v
            rho(float): Autocorrelation of x
            init(tuple, optional): Initial values of(x, y). Defaults to(0, 0).

    Returns:
            float: Bias of the OLS estimator
    """
    from numpy.random import multivariate_normal

    # Simulate errors
    u, v = multivariate_normal(
        [0, 0], [[sigma_v**2, sigma_uv], [sigma_uv, sigma_v**2]], nobs
    ).T
    # Construct the variables
    x = np.zeros(nobs) + init[0]  # Initialize x
    y = np.zeros(nobs) + init[1]  # Initialize y
    for t in range(1, nobs):
        x[t] = rho * x[t - 1] + v[t]  # x(t) = rho*x(t-1) + v(t)
        y[t] = beta * x[t - 1] + u[t]  # y(t) = beta*x(t-1) + u(t)
    # Estimate the coefficients
    rho_hat = np.corrcoef(x[1:], x[:-1])[0, 1]  # Estimate rho
    beta_hat = np.cov(y[1:], x[:-1])[0, 1] / np.var(x[:-1])  # Estimate beta
    return beta_hat - beta


def calc_bias(nreps, nobs, sigma_uv, rho):
    bias = np.array([simulate_model(nobs, sigma_uv, rho) for _ in range(nreps)])
    return bias.mean()


T_list = np.arange(50, 1001, 50)  # List of sample sizes
rho_list = [0, 0.3, 0.5, 0.9, 0.95, 0.99]  # List of autocorrelations
sigma_uv_list = np.arange(-0.9, 1, 0.1).round(1)  # List of covariances
nreps = 1000  # Number of replications

# Calculate the bias for
bias = np.array(
    [
        [
            [calc_bias(nreps, T, sigma_uv, rho) for T in T_list]
            for sigma_uv in sigma_uv_list
        ]
        for rho in rho_list
    ]
)
bias = np.abs(bias)
# bias(rho, sigma_uv, T)

# Plot the heatmap
fig, ax = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
for i, ax in enumerate(ax.flat):
    sns.heatmap(
        ax=ax,
        data=bias[i, :, :],
        vmin=0,
        vmax=bias.max(),
        square=True,
        linewidth=0,
        cmap="Purples",
        cbar=i == 0,
        cbar_ax=None if i else cbar_ax,
    )
    if i % 3 == 0:
        ax.set_ylabel(r"$\sigma_{uv}$", fontsize=14)
    if i >= 3:
        ax.set_xlabel("T", fontsize=14)
    ax.set_xticks(range(0, len(T_list), 2), T_list[::2], rotation=90)
    ax.set_yticks(range(0, len(sigma_uv_list), 2), sigma_uv_list[::2], rotation=0)
    ax.set_title(r"$\rho = %.2f$" % rho_list[i])
plt.suptitle("Absolute bias", fontsize=16)
fig.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("images/bias.jpg", dpi=1200, bbox_inches="tight")
