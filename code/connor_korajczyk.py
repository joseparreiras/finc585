import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from statsmodels.api import OLS, add_constant

nu_purple = "#492c7f"

sns.set_theme(context = "notebook",style = "whitegrid", palette = "colorblind", font_scale = 1.25, rc = {"figure.figsize": (12, 5)})
# Blue to White to Blue cmap for correlation plot
cmap = colors.LinearSegmentedColormap.from_list("blueWhiteBlue", ["C0", "white", "C0"])


# Data -------------------------------------------------------------------------

fund_returns = pd.read_csv('../data/fund_returns.csv') # Funds monthly return
# Convert to datetime
fund_returns['date'] = pd.to_datetime(fund_returns['date'], format = '%Y%m') 
fund_returns['logret'] = np.log(1+fund_returns['return']) # Log return
# Change from long to wide format
fund_returns = fund_returns.pivot_table(
    index = 'date', columns = 'fundno', values = 'logret')
fund_returns.dropna(axis = 1, inplace = True) # Drop funds with missing values

ffdaily = pd.read_csv('../data/ffdaily.csv') # Fama-French daily data
# Convert to datetime and set as index
ffdaily['date']= pd.to_datetime(ffdaily['date'], format = '%Y-%m-%d') 
ffdaily.set_index('date', inplace = True, drop = True)
risk_free = ffdaily['rf'].resample('MS').sum() # Monthly risk-free rate
ffdaily.drop(columns = 'rf', inplace = True) # Drop risk-free rate

# Calculate funds excess returns
fund_returns = fund_returns.apply(lambda x: x - risk_free, axis = 0)
fund_returns.dropna(inplace = True)

# Functions --------------------------------------------------------------------

class CK_method:
    def __init__(self, returns):
        """
        Perform Connor & Korajczyk (1983) method to evaluate mutual fund performance.

        Args:
            returns (pd.DataFrame): Dataframe of log-returns
        """
        self.nfunds = returns.shape[1] # Number of funds
        self.returns = returns # Log-returns
    
    def fit(self, nfactors):
        """
        Fits the Principal Component Analysis

        Args:
            nfactors (int): Number of principal components to calculate
        """
        from numpy.linalg import eig
        from statsmodels.api import OLS, add_constant
        fundno = self.returns.columns # Fund identifiers
        nfunds = len(fundno) # Number of funds
        Rn = self.returns.values # Returns matrix
        Omega = Rn @ Rn.T / nfunds # Cross-product matrix of returns
        eigval, eigvec = eig(Omega) # Eigenvalues and eigenvectors
        # Add explained variance and explained variance ratio to object
        self.explained_variance = eigval[:nfactors]
        self.explained_variance_ratio = sum(self.explained_variance) / sum(eigval)
        # Calculate factors as the first nfactors eigenvectors
        self.factors = pd.DataFrame(eigvec[:, :nfactors], index = self.returns.index, columns = [f'pc{i+1}' for i in range(nfactors)])
        # Fit OLS model for loadings
        self.model = OLS(Rn, add_constant(self.factors)).fit()
        # Extract alpha, loadings and idiosyncratic variance
        self.alpha = pd.Series(
            self.model.params.iloc[0,:].values, index = fundno)
        self.loadings = pd.DataFrame(
            self.model.params.iloc[1:].T.values,
            index = fundno, columns = [f'pc{i+1}' for i in range(nfactors)])
        self.idio_var = pd.Series(
            np.std(self.model.resid, axis = 0).values, index = fundno)
        
    def params(self):
        """
        Prints the model parameters

        Returns:
            tuple: Alpha, Loadings and Idiosyncratic Variance
        """
        return self.alpha, self.loadings, self.idio_var
    
    def appraisal_ratio(self):
        """
        Calculate Traynor & Black's Appraisal Ratio

        Returns:
            float: Appraisal Ratio
        """
        return self.alpha / self.idio_var

    def alpha_vcov(self):
        """
        Calculates variance of Jensen's alpha following Connot & Korajczyk (1983)
        
        Returns:
            pd.Series: Variance
        """
        tobs = self.returns.shape[0] # Size of data
        gamma = np.mean(self.factors, axis = 0) # Factor mean
        sigma = self.idio_var # Funds idiosyncratic variance
        return (1 + gamma @ gamma.T) * sigma / tobs
    
    def appraisal_var(self):
        """
        Calculate appraisal ratio variance following Connor & Korajczyk (1983), unique to all funds

        Returns:
            float: Variance
        """
        tobs = self.returns.shape[0] # Size of data
        gamma = np.mean(self.factors, axis = 0) # Factor mean
        return (1 + gamma @ gamma.T) / tobs
    
def find_factor_number(ck, maxfactors = 10, alpha = 0.05):
    """
    Applies Connor & Korajczyk (1983) method to find the optimal number of factors in an approximate factor model.

    Args:
        ck (CK_Method): Base model
        maxfactors (int, optional): Maximum number of factors to test for. Defaults to 10.
        alpha (float, optional): Size of the test. Defaults to 0.05.

    Returns:
        int: Optimal number of factors
    """
    from scipy.stats import norm
    cval = norm.ppf(1-alpha) # Normal critical value
    tobs = ck.returns.shape[0] # Number of dates
    nobs = ck.returns.shape[1] # Number of funds
    k = 1 # Start counter
    while k < maxfactors:
        ck.fit(nfactors = k) # Fit model with k factors
        eps = ck.model.resid # Calculate residuals
        sigma = eps**2/(1-(k+1)/tobs-k/nobs) # Corrected sq. resid.
        mu = (sigma).mean(axis = 1) # Average squared residuals 
        ck.fit(nfactors = k+1) # Fit model with k+1 factors
        eps_ast = ck.model.resid # Idiosyncratic variance
        sigma_ast = eps_ast**2/(1-(k+2)/tobs-(k+1)/nobs) # Corrected sq. resid.
        mu_ast = (sigma_ast).mean(axis = 1) # Average squared residuals
        delta = mu[::2].values - mu_ast[1::2].values # CK Delta
        delta_bar = delta.mean() # Average Delta
        gamma = np.sum((delta-delta_bar)**2)/(tobs/2-1) # Variance of Delta
        # Test whether mean(Delta) = 0, i.e. additional factor does not improve
        test_stat = np.sqrt(nobs)*delta_bar/np.sqrt(gamma)
        if test_stat < cval:
            # If null is not rejected, optimal number of factors is found
            break
        else:
            # If null is rejected, increase number of factors
            k += 1
    print(f'Optimal number of factors: {k}')
    return k

# Exercise ---------------------------------------------------------------------

ck = CK_method(fund_returns) # Initiate model

explained_variance = [] # Explained variance
k_list = list(range(1,9)) # Number of factors to test
for k in k_list:
    ck.fit(nfactors = k) # Fit model
    explained_variance.append(ck.explained_variance_ratio) # Append explained variance
    if k == max(k_list):
        components = ck.factors # Save components

# Calculate correlation matrix between PCA and FF
allfactors = pd.concat([ffdaily, components], axis = 1)
corr_matrix = allfactors.corr().loc[ffdaily.columns, components.columns] 

fig, ax = plt.subplots()
sns.heatmap(corr_matrix, cmap = cmap, center = 0, ax = ax, annot = True, vmin = -1, vmax = 1, fmt = '.2f', cbar = False)
ax.set_xticklabels([x.upper() for x in components.columns], rotation = 0)
ax.set_yticklabels([x.upper() for x in ffdaily.columns], rotation = 0)
plt.savefig('../images/ck_pca_ff_corr.png', dpi = 300, bbox_inches = 'tight')

# Use CK (1993) paper to find the optimal number of factors
k_star = find_factor_number(ck, maxfactors = 50, alpha = 0.05)
ck.fit(nfactors = k_star)

# Plot explained variance as a function of number of factors
fig, ax = plt.subplots()
ax.bar(k_list, explained_variance, color = nu_purple)
if k_star in k_list:
    ax.bar(k_star, ck.explained_variance_ratio, color = 'gold')
else:
    ax.bar(max(k_list) + 1, ck.explained_variance_ratio, color = 'C1')
    ax.set_xticks(k_list + [max(k_list) + 1])
    ax.set_xticklabels(k_list + [k_star])
plt.savefig('../images/ck_explained_variance.png', dpi = 300, bbox_inches = 'tight')

# Calculate appraisal ratio
appraisal = ck.appraisal_ratio()
alpha = 0.05
crit = sp.stats.norm.ppf(1-alpha)*np.sqrt(ck.appraisal_var()) # Critical value
# Plot distribution of appraisal ratio
fig, ax = plt.subplots(figsize = (12, 5))
sns.kdeplot(appraisal, color = nu_purple, ax = ax, fill = True)
# Add lines with critical values
ax.axvline(-crit, color = 'black', linestyle = '--')
ax.axvline(crit, color = 'black', linestyle = '--')
plt.savefig('../images/appraisal_ratio_dist.png')

# Subsamples 
nbins = 3 # Number of subsamples
datebins = pd.qcut(fund_returns.index, nbins, labels = False) # Split dates into bins
ck_sub = fund_returns.groupby(datebins).apply(CK_method) # Apply CK method to each subsamples

for i in range(nbins):
    # Find number of factors for each subsample
    k_star = find_factor_number(ck_sub[i], maxfactors = 50, alpha = 0.05)
    ck_sub[i].fit(nfactors = k_star)

# Calculate appraisal ratio for each subsample
alpha = 0.05
appraisal_sub = {i: ck_sub[i].appraisal_ratio() for i in range(nbins)}
appraisal_var = {i: ck_sub[i].appraisal_var() for i in range(nbins)}
crit = sp.stats.norm.ppf(1-alpha)
# Get number of funds that outperform in each subsample 
best_funds = {i: x.index[np.where(x > crit*np.sqrt(appraisal_var[i]))[0]] for i, x in appraisal_sub.items()}
# Check how many funds outperform in both samples
fund_overlap = set.intersection(*[set(best_funds[i]) for i in best_funds])
print(f'Number of Funds Overperforming in Both Subsamples: {len(fund_overlap)} ({len(fund_overlap)/fund_returns.shape[1]*100:.2f}%)')

# Plot the distribution of appraisal ratio for both subsamples
fig, ax = plt.subplots(1, nbins,  figsize = (6*nbins, 5))
fig.tight_layout()
for i in range(nbins):
    sns.kdeplot(appraisal_sub[i], ax = ax[i], color = f'C{i+1}', fill = True)
    ax[i].axvline(-crit*np.sqrt(appraisal_var[i]), color = 'black', linestyle = '--')
    ax[i].axvline(crit*np.sqrt(appraisal_var[i]), color = 'black', linestyle = '--')
    ax[i].set_title(f'Subsample {i+1}')
    ax[i].set_ylabel('')
plt.savefig('../images/apprisal_ratio_sub.png')