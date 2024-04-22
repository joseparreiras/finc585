import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from statsmodels.api import OLS, add_constant

nu_purple = "#492c7f"

sns.set_theme(context = "notebook",style = "whitegrid", palette = "colorblind", font_scale = 1.25, rc = {"figure.figsize": (12, 5)})

data = pd.read_csv("../data/stocks_pca_large.zip", compression = "zip")
data.rename({x: x.lower() for x in data.columns}, axis = 1, inplace = True)
data['date'] = pd.to_datetime(data['date'], format = '%d%b%Y')
data = data.pivot_table(index = "date", columns = "permno", values = "price")
returns = np.log(data).diff()[1:]
returns.dropna(axis = 1, inplace = True)

ffdaily = pd.read_csv("../data/ffdaily.csv")
ffdaily['date'] = pd.to_datetime(ffdaily['date'], format = '%Y-%m-%d')
ffdaily.set_index('date', inplace = True, drop = True)
ffdaily = ffdaily[ffdaily.index.isin(returns.index)]

returns -= ffdaily['rf'].values.reshape(-1,1) # Calculate excess returns
ffdaily.drop('rf', axis = 1, inplace = True) # Remove risk free data

# Step 1: PCA Analysis
num_factors = range(1, 9)
explained_variances = []
pca_components = []
bic = []
tobs = returns.shape[0]
scaler = StandardScaler() # Set scaler function: (x - mean)/std

for n in num_factors:    
    # Standardize Returns
    standard_returns = scaler.fit_transform(returns)
    # Run PCA
    pca = PCA(n_components=n)
    pca.fit(standard_returns.T)
    explained_variances.append(sum(pca.explained_variance_ratio_))
    pca_components.append(pd.DataFrame(pca.components_.T, index = returns.index, columns = [f"pc{i+1}" for i in range(n)]))
    bic.append(tobs*np.log(pca.noise_variance_)+n*np.log(tobs))

fig, ax = plt.subplots(figsize = (12, 5))
ax.plot(pca_components[0])
plt.savefig("../images/pca_first_component.png", dpi = 300, bbox_inches = 'tight')
fig, ax = plt.subplots(1, 2, figsize = (20, 5))
# fig.tight_layout()
ax[0].bar(num_factors, [x*100 for x in explained_variances])
ax[0].set_xlabel("Number of Factors")
ax[0].set_ylabel("Explained Variance (%)")
ax[1].bar(num_factors, bic)
ax[1].set_yscale('log')
ax[1].set_xlabel("Number of Factors")
ax[1].set_ylabel("Bayes Information Criteria (log)")
plt.savefig('../images/pca_explained_variance.png', dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(4,2,figsize = (12*2,5*4), sharey = True)
fig.tight_layout()
for i, x in enumerate(ax.flatten()):
    x.plot(pca_components[-1].iloc[:,i])
    x.set_title(f"Component {i+1}")
    x.set_xlim(returns.index.min(), returns.index.max())
plt.savefig("../images/pca_components.png", dpi = 300, bbox_inches = 'tight')   

# Step 2: Factor Rotation
rsquared = {}
fitted = pd.DataFrame()
for factor in ffdaily.columns:
    model = OLS(ffdaily[factor], add_constant(pca_components[-1])).fit()
    rsquared.update({factor: model.rsquared})
    print(model.f_pvalue)
    fitted[factor] = model.fittedvalues

fig, ax = plt.subplots(figsize = (12, 5))
fig.tight_layout()
ax.bar(rsquared.keys(), rsquared.values())
ax.set_xlabel("Factor")
ax.set_ylabel("R-Squared")
plt.savefig("../images/pca_ffdaily_r2.png", dpi = 300, bbox_inches = 'tight')

corr_matrix = pd.merge(ffdaily, pca_components[-1], left_index = True, right_index = True).corr().loc[ffdaily.columns, pca_components[-1].columns]
corr_matrix.rename({x: x.upper() for x in corr_matrix.index}, axis = 0, inplace = True)
corr_matrix.rename({x: x.upper() for x in corr_matrix.columns}, axis = 1, inplace = True)

fig, ax = plt.subplots(figsize = (12, 5))
sns.heatmap(corr_matrix.round(2), cmap = "RdBu", ax = ax, vmax = 1, vmin = -1, annot = True)
ax.set_title("Correlation Between FF5 and PCA Factors")
plt.savefig("../images/pca_ffdaily_corr.png", dpi = 300, bbox_inches = 'tight')

# Step 3: Canonical Correlation Analysis
num_components = range(1, 6)
cca_score = []
for n in num_components:
    cca = CCA(n_components = n)
    cca.fit(pca_components[-1], ffdaily)
    cca_score.append(cca.score(pca_components[-1], ffdaily))

fig, ax = plt.subplots(figsize = (12, 5))
fig.tight_layout()
ax.bar(num_components, cca_score)
ax.set_xlabel("Number of Components")
ax.set_ylabel("Canonical Correlation")
plt.savefig("../images/cca_score.png", dpi = 300, bbox_inches = 'tight')

n_optimal = np.argmax(cca_score) + 1
cca = CCA(n_components = n_optimal)
cca.fit(pca_components[-1], ffdaily)
X_c, Y_c = cca.transform(pca_components[-1], ffdaily)

fig, ax = plt.subplots(n_optimal, 1, figsize = (12, 15))
for i in range(n_optimal):
    sns.regplot(X_c[:,i], Y_c[:,i], ax = ax[i], ci=None, line_kws = {'color': 'orange'}, scatter_kws = {'color': nu_purple, 'alpha': 0.5})
    ax[i].set_title(f"#{i+1} Pair of Canonical Variables")
    ax[i].legend()
plt.savefig("../images/canonical_variables_scatter.png", dpi = 300, bbox_inches = 'tight')

corr_matrix = np.corrcoef(X_c, Y_c, rowvar = False).round(2)
fig, ax = plt.subplots(figsize = (12, 5))
sns.heatmap(corr_matrix[n_optimal:, :n_optimal], cmap = "RdBu", ax = ax, vmax = 1, vmin = -1, annot = True)
ax.set_xlabel("Canonical Variables: PCA Factors")
ax.set_ylabel("Canonical Variables: FF5 Factors")
plt.savefig("../images/cca_corr.png", dpi = 300, bbox_inches = 'tight')