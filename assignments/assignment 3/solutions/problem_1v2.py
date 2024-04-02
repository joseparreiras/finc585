"""
FINC 585-3: Asset Pricing
Assignment 3 (Solutions)
Prof. Torben Andersen
TA: Jose Antunes-Neto
"""

# Modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
import seaborn as sns
from scipy.stats import chi2
from statsmodels.sandbox.regression import gmm
from numba import njit

nupurple = "#492C7F"

# Data -------------------------------------------------------------------------

data = pd.read_excel("data/HansenSingletondata.xlsx")
data["returns"] = data["sp500"] / data["sp500"].shift(1)
data["cgrowth"] = data["Consumption"] / data["Consumption"].shift(1)
data.index = pd.date_range("1958-01-01", "2017-01-01", freq="Q")
data = data.iloc[1:, 2:]

# Plotting the R and C values as in the problem set (for comparison)
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax[0].plot(data.returns, label=r"$R_{t+1}^m$", color="darkblue")
ax[0].axhline(1, linestyle="--", color="black")
ax[0].set_ylim([0.7, 1.4])
ax[0].legend()
ax[1].plot(data.cgrowth, label=r"$C_{t+1}/C_t$", color="orange")
ax[1].axhline(1, linestyle="--", color="black")
ax[1].set_ylim([0.95, 1.05])
ax[1].legend(frameon=False)
# plt.savefig("images/series.jpg", dpi=300, bbox_inches="tight")

# Class ------------------------------------------------------------------------


def transform(x):
    return np.array([1 / (1 + np.exp(-x[0])), np.exp(x[1])])


def invert(x):
    return np.array([np.log(x[0] / (1 - x[0])), np.log(x[1])])


class HansenSingleton:
    def __init__(self, data):
        self.data = data

    def m(self, beta, gamma, inst=None):
        if inst is None:
            inst = np.ones((self.data.shape[0], 1))
        R, C = self.data.values.T
        beta, gamma = transform((beta, gamma))
        return (R - beta * (C ** (-gamma)) - 1).reshape(-1, 1) * inst

    def g(self, beta, gamma, inst=None):
        return self.m(beta, gamma, inst).mean(axis=0)

    def G(self, beta, gamma, inst=None):
        if inst is None:
            inst = np.ones((self.data.shape[0], 1))
        R, C = self.data.values.T
        beta, gamma = transform((beta, gamma))
        deriv_beta = np.array((R * C ** (-gamma)).reshape(-1, 1) * inst)
        deriv_gamma = np.array(
            (-beta * R * C ** (-gamma) * np.log(C)).reshape(-1, 1) * inst
        )
        deriv = np.vstack((deriv_beta.mean(axis=0), deriv_gamma.mean(axis=0)))
        return deriv


def fitGMM(mom, params_init, weight_matrix=None, **kwargs):
    from scipy.optimize import minimize

    q = mom(params_init).shape[1]
    if weight_matrix is None:
        weight_matrix = np.identity(q)
    g = lambda params: mom(params).mean(axis=0)
    solver = minimize(
        lambda params: np.dot(g(params).T, np.dot(weight_matrix, g(params))),
        x0=params_init,
        **kwargs
    )
    if not solver.success:
        raise ValueError("Solver did not converge")
    else:
        estimate = solver.x
    return solver


def calc_vcov(mom, params, method="NW", maxlags=4):
    m = mom(params)
    nobs = m.shape[0]

    def gamma(lag):
        return 1 / nobs * np.dot(m[:-lag].T, m[lag:])

    vcov = 1 / nobs * np.dot(m.T, m)
    if method == "IID":
        return vcov
    for j in range(1, maxlags + 1):
        if method == "NW":
            c = 1 - j / (maxlags + 1)
        elif method == "Hansen":
            c = 1
        vcov += 2 * c * gamma(j)
    return vcov


model = HansenSingleton(data[1:])
inst = data.shift(1)[1:]
inst["constant"] = np.ones(inst.shape[0])
inst = inst.values

mom = lambda params: model.m(*params, inst)

# Continuous Update GMM
eps = 1e-12
maxiter = 1000

p0 = invert((0.999, 0.63))
j, err = 0, 1
while j < maxiter and err > eps:
    S = calc_vcov(mom, p0, method="IID", maxlags=6)
    weight_matrix = np.linalg.inv(S)
    solver = fitGMM(
        mom=lambda params: model.m(*params, inst),
        params_init=p0,
        weight_matrix=weight_matrix,
    )
    p1 = solver.x
    err = np.max(np.abs(p1 - p0))
    print((j, err))
    print(transform(p1))
    p0 = p1
    j += 1

theta = transform(p1)


# Calculate the standard errors
def GMMvcov_efficient(model, params, inst, weight_matrix):
    nobs = model.data.shape[0]
    G = model.G(*params, inst)
    return 1 / nobs * np.linalg.inv(np.dot(G, np.dot(weight_matrix, G.T)))


theta_vcov = GMMvcov_efficient(model, invert(theta), inst, np.linalg.inv(S))
theta_std = np.sqrt(np.diag(theta_vcov))

theta


def GMMvcov(model, params, inst, weight_matrix, S):
    nobs = model.data.shape[0]
    G = model.G(*params, inst)
    bread = np.dot(G, np.dot(weight_matrix, G.T))
    meat = np.dot(G, np.dot(weight_matrix, np.dot(S, np.dot(weight_matrix.T, G.T))))
    return 1 / nobs * np.dot(np.linalg.inv(bread), np.dot(meat, np.linalg.inv(bread)))
