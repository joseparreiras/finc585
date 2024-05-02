import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from statsmodels.api import OLS, add_constant

nu_purple = "#492c7f"

sns.set_theme(context = "notebook",style = "whitegrid", palette = "colorblind", font_scale = 1.25, rc = {"figure.figsize": (12, 5)})
# Create a blue-white-blue colormap
cmap = colors.LinearSegmentedColormap.from_list("blueWhiteBlue", ["C0", "white", "C0"])


# Data -------------------------------------------------------------------------

ffdaily = pd.read_csv('../data/ffdaily.csv', index_col=0, parse_dates=True)
industry = pd.read_csv('../data/industry_daily.csv', index_col=0, parse_dates=True)

data = pd.merge(ffdaily, industry, left_index = True, right_index = True)
industry = data.loc[:, industry.columns]
ffdaily = data.loc[:, ffdaily.columns]

industry = industry.apply(lambda x: x - ffdaily['rf'], axis = 0)
ffdaily = ffdaily.drop('rf', axis = 1)

# PCA Analysis -----------------------------------------------------------------

class PCAnalysis:
    def __init__(self, data):
        from sklearn.preprocessing import StandardScaler as scaler
        self.data = scaler().fit_transform(data)
        self.original_data = data
        
    def fit(self, nfactors):
        from sklearn.preprocessing import StandardScaler as scaler
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components = nfactors)
        self.factors = self.pca.fit_transform(self.data)
        self.nfactors = nfactors
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.explained_variance = self.pca.explained_variance_
        
    def loadings(self):
        from statsmodels.api import OLS, add_constant
        self.ols = OLS(self.data, add_constant(self.factors)).fit()
        return self.ols.params[1:]
    
    def resid(self):
        return self.ols.resid
    
    def bic(self):
        from scipy.stats import multivariate_normal
        if not hasattr(self, 'ols'):
            self.loadings()
        k = self.nfactors
        tobs = self.data.shape[0]
        nobs = self.data.shape[1]
        Sigma = np.cov(self.resid().T)        
        loglik = np.sum(multivariate_normal.logpdf(
            self.resid(), cov = Sigma, allow_singular = True))
        return k*np.log(tobs*nobs) - 2*loglik
    
    def aic(self):
        from scipy.stats import multivariate_normal
        if not hasattr(self, 'ols'):
            self.loadings()
        k = self.nfactors
        tobs = self.data.shape[0]
        nobs = self.data.shape[1]
        Sigma = np.cov(self.resid().T)
        loglik = np.sum(multivariate_normal.logpdf(
            self.resid(), cov = Sigma, allow_singular = True))
        return 2*k - 2*loglik

num_factors = range(1, 11)
explained_variance = []
aic = []
bic = []

for k in num_factors:
    pca = PCAnalysis(industry)
    pca.fit(k)
    explained_variance.append(sum(pca.explained_variance_ratio))
    aic.append(pca.aic())
    bic.append(pca.bic())
    if k == max(num_factors):
        components = pd.DataFrame(
            pca.factors, index = industry.index, columns = [f"PC{i+1}" for i in range(k)])


fig, ax = plt.subplots(1, 3, figsize = (12*3, 5))
ax[0].bar(num_factors, explained_variance)
ax[0].set_title("Explained Variance")
ax[0].set_xlabel("Number of Factors")
ax[1].bar(num_factors, aic)
ax[1].bar(num_factors[np.argmax(aic)], max(aic), color = 'C1')
ax[1].set_title("Akaike Information Criteria")
ax[1].set_xlabel("Number of Factors")
ax[2].bar(num_factors, bic)
ax[2].bar(num_factors[np.argmax(bic)], max(bic), color = 'C1')
ax[2].set_title("Bayesian Information Criteria")
ax[2].set_xlabel("Number of Factors")
plt.savefig('../images/pca_criteria.png', dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots()
ax.plot(pd.Series(components.iloc[:,0], index = industry.index), label = 'PC1')
ax2 = ax.twinx()
ax2.plot(ffdaily['mkt-rf'], color = 'C1', label = 'Market')
ax2.grid(False)
# Get the legend handles and labels from ax and ax2
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine the handles and labels
handles = handles1 + handles2
labels = labels1 + labels2
# Create a single legend with the combined handles and labels
ax.legend(handles, labels, loc='upper left', ncol = 2)
plt.savefig('../images/pca_vs_market.png', dpi = 300, bbox_inches = 'tight')

all_factors = pd.concat([ffdaily, components], axis = 1)
factor_corr = all_factors.corr().loc[ffdaily.columns, components.columns]

fig, ax = plt.subplots()
sns.heatmap(factor_corr, cmap = cmap, center = 0, ax = ax, annot = True, vmin = -1, vmax = 1, cbar = False)
ax.set_yticklabels([x.upper() for x in ffdaily.columns], rotation = 0)
plt.savefig('../images/factor_corr.png', dpi = 300, bbox_inches = 'tight')


fig, ax = plt.subplots(figsize = (20, 5))
sns.heatmap(pca.loadings(), cmap = cmap, center = 0, ax = ax)
ax.set_xticklabels([x.title() for x in industry.columns], rotation = 90, ha = 'center')
ax.set_yticklabels([f"PC{i+1}" for i in range(pca.nfactors)], rotation = 0)
plt.savefig('../images/factor_loadings.png', dpi = 300, bbox_inches = 'tight')

# CCA 
ncomponents = range(1, 6)
cca_score = []
for n in ncomponents:
    cca = CCA(n_components = n)
    cca.fit(components, ffdaily.values)
    cca_score.append(cca.score(components, ffdaily.values))
    
fig, ax = plt.subplots()
ax.bar(ncomponents, cca_score)
ax.set_xlabel("Number of Components")
ax.set_ylabel("Canonical Correlation")
plt.savefig('../images/cca_score.png', dpi = 300, bbox_inches = 'tight')