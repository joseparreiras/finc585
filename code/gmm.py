"""
FINC 585-3: Asset Pricing
Assignment 3 (Solutions)
Prof. Torben Andersen & Zhengyang Jiang
TA: Jose Antunes-Neto

This code solves the GMM estimation for the Hansen-Singleton model. The data is made available in the beginning of the course, but it can also be found on the `data/` folder. For more information, check the `README.md` file.
For the estimation, a custom GMM class is provided under the `gmm` class. This is used to estimate the model and calculate the necessary statistics.
This code does not provides estimation for the robust variance covariance matrices, as it is still under development.
FIXME: This code is incomplete and needs to be finished.
"""

# Modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
import seaborn as sns
from scipy.stats import chi2

nupurple = "#492C7F"

# Set seaborn theme for plots
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="colorblind",
    font="sans-serif",
    font_scale=1.25,
)


# Data -------------------------------------------------------------------------

data = pd.read_excel("../data/HansenSingletondata.xlsx")
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
plt.savefig("../images/series.jpg", dpi=300, bbox_inches="tight")

# Class ------------------------------------------------------------------------

class gmm:
    def __init__(self, g, X, Z):
        """
        Initializes the GMM estimator.

        Args:
            g (function): A function that takes the data matrix X, the instrument matrix Z, and the coefficients b, and returns the moment conditions.
            X (array-like): The data matrix.
            Z (array-like): The instrument matrix.
        """
        self.g = g
        self.X = np.array(X)
        self.Z = np.array(Z)

    def fit(self, b0, W=None, echo=False):
        """
        Fits the GMM estimator.

        Args:
            b0 (array-like): The initial guess for the coefficients.
            W (array-like, optional): The weighting matrix. Defaults to None (identity matrix).
            echo (bool, optional): Whether to print the optimization message. Defaults to False.

        Returns:
            array-like: The estimated coefficients.
        """
        from scipy.optimize import minimize

        X, Z = self.X, self.Z
        g = self.g  # Moment function
        p = len(g(X, Z, b0).mean(axis=0))  # Number of parameters
        if W is None:
            W = np.identity(p)  # Start with identity weighting matrix
        # Solve the minimization problem
        solve = minimize(
            lambda b: g(X, Z, b).mean(axis=0).T @ W @ g(X, Z, b).mean(axis=0), x0=b0
        )
        if echo:
            print(solve.message)  # Print the optimization message
        if not solve.success:
            raise ValueError("Optimization failed.")
        else:
            self.W = W  # Store the weighting matrix
            self.coef = solve.x  # Store the estimated coefficients
            return self.coef

    def calc_vcov(self, grad, efficient=False):
        """
        Calculates the variance-covariance matrix of the estimated coefficients.

        Args:
            grad (function): A function that takes the data matrix X, the instrument matrix Z, and the coefficients b, and returns the gradient of the moment conditions.
            efficient (bool, optional): Whether to use the efficient variance-covariance matrix. Defaults to False.

        Returns:
            array-like: The variance-covariance matrix of the estimated coefficients.
        """
        X, Z = self.X, self.Z
        g = self.g  # Moment function
        n = X.shape[0]  # Number of observations
        coef = self.coef  # Estimated coefficients
        W = self.W  # Weighting matrix
        G = grad(X, Z, coef)  # Gradient of the moment function
        if not efficient:
            # If not efficient, calculate bread-meat-bread sandwich
            # Covariance matrix of the moment function
            sigma = np.cov(g(X, Z, coef).T)
            # Inverse of the Hessian matrix
            hess_inv = np.linalg.inv(G.T @ W @ G)
            # Variance-covariance matrix
            vcov = 1 / n * hess_inv @ (G.T @ W @ sigma @ W.T @ G) @ hess_inv
        else:
            # If efficient, calculate the efficient variance-covariance matrix
            vcov = 1 / n * np.linalg.inv(G.T @ W @ G)
        self.vcov = vcov  # Store the variance-covariance matrix
        return vcov


# GMM --------------------------------------------------------------------------


# (a)
"""
In this question we will examine manually the J-criterion for the unconditional moment and find an initial guess for where should our parameters be. We will fix beta at 0.99 and plot the J-criterion for different values of gamma.
"""

gamma_range = np.linspace(0, 5, 1000)  # Range of gamma values
beta = 0.99  # Set default beta value


def g(X, gamma):
    # Unconditional moment function
    X = np.array(X)
    g = X[:, 0] * beta * X[:, 1] ** (-gamma) - 1  # R*beta*u'(c)-1
    return g.reshape(-1, 1)


# Plotting the unconditional moment function for the grid of gamma values
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(
    gamma_range, [g(data, gamma).mean() ** 2 for gamma in gamma_range], color=nupurple
)
ax.set_xlabel(r"$\gamma$")
plt.savefig("../images/jcrit.jpg", dpi=300, bbox_inches="tight")

# Save the initial guess for gamma
gamma0 = gamma_range[np.argmin([g(data, gamma).mean() ** 2 for gamma in gamma_range])]

# (b)
"""
We now turn to using conditional moments to improve this first analysis. Since the moment condition is given on a conditional basis, we can use any lagged value of (R,C) to improve our estimation. We will use the lagged values of the data to construct the instrument matrix. We will also add a constant to the instrument matrix and plot a similar figure as in (a).
"""

gamma_range = np.linspace(0, 5, 1000)  # Range of gamma values
inst = data.shift(1)[1:]  # Instrument matrix
inst["const"] = 1  # Adding a constant
inst = np.array(inst)  # Converting to array

# Plotting
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(
    gamma_range,
    [(g(data[1:], gamma) * inst).mean(axis=0) ** 2 for gamma in gamma_range],
    label=[r"$R_{t+1}^m$", r"$C_{t+1}/C_t$", "Constant"],
)
ax.legend()
ax.set_xlabel(r"$\gamma$")
plt.savefig("../images/jcrit_cond.jpg", dpi=300, bbox_inches="tight")

# (c)
"""
We now finally estimate our parameters. We will use the conditional moments since they allow our model to be just(or over)identified so we can estimate both beta and gamma.
Since gamma and beta are restricted parameters (gamma > 0) and (1 > beta > 0), I will make a transformation of these parameters that allow our algorithm to search over the real line and avoid constraints. The transformation is as follows:
beta' = 1/(1+exp(beta))
gamma' = gamma^2
"""


def transform_pars(b):
    # Reparametrization of the parameters
    return (1 / (1 + np.exp(-b[0])), b[1] ** 2)


def invert_pars(b):
    # Function to revert the reparametrization
    return (np.log(b[0] / (1 - b[0])), np.sqrt(b[1]))


def g(X, params):
    # Unconditional moment function
    beta, gamma = transform_pars(params)
    X = np.array(X)
    g = X[:, 0] * beta * X[:, 1] ** (-gamma)  # R*beta*u'(c)-1
    return g.reshape(-1, 1)


def g(X, Z, b):
    # Conditional moment function
    X = np.array(X)  # (R,C) data
    Z = np.array(Z)  # Instrument matrix
    # Computing the moment function
    b = transform_pars(b)  # Transforming the parameters
    uncond = np.array(X[:, 0] * b[0] * X[:, 1] ** (-b[1]) - 1)  # R*beta*u'(c)-1
    # (R*beta*u'(c)-1) * Z
    return np.array([np.multiply(uncond, z) for z in Z.T]).T


"""
The estimation takes two steps. We start with an initial guess of (beta, gamma) and estimate the model using an identity weighting matrix as a first step and store the estimates. These are used on a second step to estimate the efficient weighting matrix and reestimate the model.
"""

# Initial guess (we need to invert so the transformation keeps (0.99, gamma0) as initial guess)
b0 = invert_pars((0.99, gamma0))
hs = gmm(g, data[1:], inst)  # Initialize the GMM class
theta_tilde = transform_pars(hs.fit(b0=b0))  # First step estimation
n = data.shape[0] - 1  # Sample size
# Estimate the S matrix (second moment of the moment conditions)
S_tilde = S(data[1:], inst, hs.coef)
# Study the eigenvalues of S_tilde
S_eigvals = np.linalg.eigvals(S_tilde)

"""
We can see that the minimum eigenvalue is very low, therefore we would likely be in trouble if we used the inverse of the S matrix to calculate the efficient weighting matrix.
Let's just try it for the sake of the exercise
"""
W_tilde = np.linalg.inv(S_tilde)  # Efficient weighting matrix
# Second step estimation
theta_hat = transform_pars(hs.fit(b0=hs.coef, W=W_tilde))
print(theta_hat)
"""
As we can see from the results, the estimates of gamma are not reasonable. That happens because W_tilde is probably not a good approximation to the optimal weighting matrix in this sample. As we discussed, this happens because the matrix S is nearly singular. I will use the first step estimate as our final results.
"""
theta_hat = transform_pars(hs.fit(b0=b0))  # First step estimation

# Calculating the variance matrix


def G(X, Z, b):
    # Gradient of the moment function for VCOV estimation
    X = np.array(X)  # (R,C) data
    Z = np.array(Z)  # Instrument matrix
    grad = np.array(
        # dg/dbeta
        [
            ((X[:, 0] * b[0] * X[:, 1] ** (-b[1])).reshape(-1, 1) * Z).mean(axis=0),
            # dg/dgamma
            (
                (-X[:, 0] * b[0] * X[:, 1] ** (-b[1]) * np.log(X[:, 1])).reshape(-1, 1)
                * Z
            ).mean(axis=0),
        ]
    )
    return grad.T


vcov_theta = hs.calc_vcov(G, efficient=False)  # Estimate the variance matrix

# Exporting the results to latex
table = "\\begin{tabular}{ccccc}"
table += "\n$\\beta$ = & %.4f & & $\\gamma$ = & %.4f \\\\\n& (%.4f) & & & (%.4f)" % (
    *theta_hat,
    *np.sqrt(np.diag(vcov_theta)),
)
table += "\n\\end{tabular}"
with open("../tables/hs_equation.tex", "w") as e:
    e.write(table)

# Hausman test
"""
We now perform the Hausman test of overidentifying restrictions.
"""


def hausman_test(X, Z, b):
    # Performs the Hausman test
    n = X.shape[0]  # Sample size
    g0 = g(X, Z, b).mean(axis=0)  # Average sample moment
    S_tilde = S(X, Z, b)  # S matrix
    h_stat = n * g0.T @ np.linalg.inv(S_tilde) @ g0  # Test statistic
    pvalue = chi2.cdf(h_stat, 1)  # p-value
    return h_stat, pvalue


hausman = hausman_test(data[1:], inst, hs.coef)  # Hausman test

# Exporting the results to latex
table = (
    "\\begin{tabular}{ccccc} \n$\\chi^2$ =& %.4f & & p-value =& %.4f \n\\end{tabular}"
    % (test_stat, pvalue)
)
with open("../tables/hs_hausman.tex", "w") as h:
    h.write(table)

# Validating local solutions
"""
I now perform a validation on the local solution by changing the initial guess. I will use a grid of 10000 points in the parameter space and estimate the model for each point. I will then plot the distribution of the estimates.
I will look for values of beta in (0,1) and gamma in (0, 15).
"""

np.random.seed(int("02011921"))  # Setting the seed
J = 10000  # Number of reps
theta_tries = list()
j = 0
while j < J:
    # Draw initial guesses
    b0 = invert_pars(np.random.rand(2) * (1, 15))
    try:
        # Estimate the model
        theta_j = transform_pars(hs.fit(b0))
        theta_tries += [theta_j]
        j += 1
    except ValueError:
        # If model does not converge, draw another
        next

theta_tries = np.array(theta_tries)

# Ploting the marginal distrbutions
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.kdeplot(ax=ax[0], x=theta_tries[:, 0], clip=(0.0, 1.0), color=nupurple)
sns.kdeplot(ax=ax[1], x=theta_tries[:, 1], clip=(0.0, 1e12), color=nupurple)
x1 = ax[0].lines[0].get_xydata()[:, 0]
y1 = ax[0].lines[0].get_xydata()[:, 1]
ax[0].fill_between(x1, y1, color=nupurple, alpha=0.3)
x1 = ax[1].lines[0].get_xydata()[:, 0]
y1 = ax[1].lines[0].get_xydata()[:, 1]
ax[1].fill_between(x1, y1, color=nupurple, alpha=0.3)
ax[1].set_ylabel("")
ax[0].set_yticks([])
ax[1].set_yticks([])
ax[0].set_title(r"$\beta$")
ax[1].set_title(r"$\gamma$")
plt.savefig("../images/marg_solutions.jpg", dpi=300, bbox_inches="tight")

# (d)
"""
We now turn to using 4 lagged values of the instruments to increase the number of moment conditions. We will use the same instruments as in the previous exercise.
"""

b0 = hs.coef  # Initial guess
# Data with 1 lag ([4:] is used to match the dimensions of ALL the arrays)
inst1 = np.array(data.shift(1))[4:]
inst1 = np.hstack((np.ones((len(inst1), 1)), inst1))  # Add constant
inst2 = np.hstack((inst1, np.array(data.shift(2))[4:]))  # Data with 2 lags
inst3 = np.hstack((inst2, np.array(data.shift(3))[4:]))  # Data with 3 lags
inst4 = np.hstack((inst3, np.array(data.shift(4))[4:]))  # Data with 4 lags


model = {
    1: gmm(g, data[4:], inst1),
    2: gmm(g, data[4:], inst2),
    3: gmm(g, data[4:], inst3),
    4: gmm(g, data[4:], inst4),
}

theta = {i: transform_pars(model[i].fit(b0=b0)) for i in model.keys()}
vcov_theta = {i: model[i].calc_vcov(G) for i in model.keys()}
hausman = {i: hausman_test(data[4:], model[i].Z, model[i].coef) for i in model.keys()}

n = data[4:].shape[0]
fmt = ".3f"
table = "\\begin{tabular}{@{\\extracolsep{5pt}}lcccc} \n\\\\[-1.8ex]\\hline \n\\hline \\\\[-1.8ex] \n& \\multicolumn{4}{c}{\\textit{Number of Lags}} \\\\ \n& (1) & (2) & (3) & (4) \\\\ \n\\hline \\\\[-1.8ex] \n$\\beta$ "
for i in model.keys():
    table += "& %s" % format(theta[i][0], fmt)
table += "\\\\ \n"
for i in model.keys():
    sd = np.sqrt(vcov_theta[i][0, 0])
    table += "& (%s)" % format(sd, fmt)
table += "\\\\ \n$\\gamma$ "
for i in model.keys():
    table += "& %s" % format(theta[i][1], fmt)
table += "\\\\ \n"
for i in model.keys():
    sd = np.sqrt(vcov_theta[i][1, 1])
    table += "& (%s)" % format(sd, fmt)
table += "\\\\ \n\\hline \\\\[-1.8ex] \nN & %i & %i & %i & %i" % (n, n, n, n)
table += "\\\\ \n $\mathcal H$ & %.3f & %.3f & %.3f & %.3f" % tuple(
    x[0] for x in hausman.values()
)
table += "\\\\ \n p-value & %.3f & %.3f & %.3f & %.3f" % tuple(
    x[1] for x in hausman.values()
)
table += " \\\\ \n\\hline \\hline \\\\[-1.8ex] \n\\end{tabular}"

with open("../tables/many_lags.tex", "w") as t:
    t.write(table)
