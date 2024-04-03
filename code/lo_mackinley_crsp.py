"""
FINC 585-3: Asset Pricing
Assignment 1 (Solutions)
Prof. Torben Andersen & Zhengyang Jiang
TA: Jose Antunes-Neto

This assignment is a solution for the Lo & MacKinlay problem in the course. The question asks you to implement the LM Variance Ratio test using the CRSP value and equal-weighted index returns.  The data is made available in the beginning of the course, but it can also be found on the `data/` folder. For more information, check the `README.md` file.
"""

# Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy as sp
import warnings
warnings.filterwarnings('ignore')

# Graphic Settings
sns.set_theme(context='notebook',style='whitegrid',palette='colorblind',font='sans-serif',font_scale=1.25)

# Import CRSP daily data
crsp = pd.read_csv('../data/CRSP_daily.csv', index_col=0)
crsp.index = pd.to_datetime(crsp.index, format = '%Y%m%d')

def vrtest(x: pd.Series, q: int, method: str = 'LM'):
    """
    This function calculates the Variance Ration test as in Lo & MacKinlay (1988).

    Args:
        x (pd.Series): Series of returns
        q (int): Maximum lag to account for
        method (str, optional): Method to calculate the asymptotic variance. Defaults to 'LM' (Lo & MacKinlay - 1988). Other option is 'HLZ' (Hong, Linton & Zhang - 2017).

    Returns:
        dict: Dictionary containing the test statistics, its standard error, and the p-value
    """
    from scipy.stats import norm
    x = x-x.mean()  # Demean the series
    T = len(x)  # Number of observations
    sigma = np.sqrt(1/T*(x**2).sum())  # Standard deviation
    rho_list = np.array([x.autocorr(j)
                        for j in range(0, q)])  # Autocorrelations
    # Variance Ratio Statistics
    vr_stat = 1+2*sum([(1-j/q)*rho_list[j] for j in range(1, q)])

    if method == 'LM':
        # L0 & MacKinlay (1988) asymptotic variance
        omega = 2*(q-1)*(2*q-1)/3*q  # Asymptotic variance
        z = np.sqrt(T*q)*(vr_stat-1)/np.sqrt(omega)  # Standardized VR
        vr_std = 1/np.sqrt(T*q)*np.sqrt(omega)  # Standard error
        pvalue = 1-norm.cdf(np.abs(z))  # P-value

    if method == 'HLZ':
        # Robust variance according to Hong, Linton & Zhang (2017)
        def chi(k, j):
            # Fourth moment of the returns
            return 1/T*np.sum(x*x.shift(j)*x*x.shift(k))
        # VR asymptotic variance
        omega = 4*np.sum([(1-j/q)*(1-k/q)*chi(k, j)/sigma **
                         4 for j in range(1, q) for k in range(1, q)])
        vr_std = 1/np.sqrt(T)*np.sqrt(omega)  # Standard error
        z = (vr_stat-1)/vr_std  # Standardized VR
        pvalue = 2*(1-norm.cdf(np.abs(z)))  # P-value

    return{
        'vr_stat': vr_stat,
        'vr_std': vr_std,
        'p_value': pvalue
    }


# Value weighted
q_max = 20 # Maximum lag
cval = sp.stats.norm.ppf(0.95) # Normal critical value


def vr_table(x, q_max=20):
    # This function creates a table with the VR test statistics for different aggregation values
    return pd.DataFrame(
        # LM Std Errors
        {q: vrtest(x, q, method='LM') for q in range(2, q_max+1)}
    ).T.join(
        # HLZ Std Errors
        pd.DataFrame(
            {q: vrtest(x, q, method='HLZ') for q in range(2, q_max+1)}
        ).T.drop('vr_stat', axis=1),
        lsuffix='_LM', rsuffix='_HLZ'
    )


# Test for Value Weighted on both subsamples
value_sample1 = vr_table(crsp['vwretd']['1991': '2006'], q_max=20)
value_sample2 = vr_table(crsp['vwretd']['2007': ], q_max=20)
# Test for Equal Weighted on both subsamples
equal_sample1 = vr_table(crsp['ewretd']['1991': '2006'], q_max=20)
equal_sample2 = vr_table(crsp['ewretd']['2007': ], q_max=20)

# Graph

# Value Weighted
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# 1991-2006
ax.plot(value_sample1['vr_stat'], label='VR Statistic',
        linewidth=1.5, marker='o')
# Confidence Interval
ax.fill_between(
    value_sample1.index,
    value_sample1['vr_stat']-cval*value_sample1['vr_std_HLZ'],
    value_sample1['vr_stat']+cval*value_sample1['vr_std_HLZ'],
    color = 'skyblue', alpha=1, label='CI (HLZ)'
)
ax.fill_between(
    value_sample1.index,
    value_sample1['vr_stat']-cval*value_sample1['vr_std_LM'],
    value_sample1['vr_stat']+cval*value_sample1['vr_std_LM'],
    color = 'skyblue', alpha=0.4, label='CI (LM)'
)
# Baseline at 1
ax.axhline(1, color='black', linewidth=.5)
ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_xlabel('Aggregation Value (q)')
fig.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(1))  # Set x-axis ticks to integers
plt.savefig('../images/vrtest_value_sample1.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# First subsample
ax.plot(value_sample2['vr_stat'], label='VR Statistic',
        linewidth=1.5, marker='o')
# Confidence Interval
ax.fill_between(
    value_sample2.index,
    value_sample2['vr_stat']-cval*value_sample2['vr_std_HLZ'],
    value_sample2['vr_stat']+cval*value_sample2['vr_std_HLZ'],
    color='skyblue', alpha=1, label='CI (HLZ)'
)
ax.fill_between(
    value_sample2.index,
    value_sample2['vr_stat']-cval*value_sample2['vr_std_LM'],
    value_sample2['vr_stat']+cval*value_sample2['vr_std_LM'],
    color='skyblue', alpha=0.4, label='CI (LM)'
)
# Baseline at 1
ax.axhline(1, color='black', linewidth=.5)
ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_xlabel('Aggregation Value (q)')
fig.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(1))  # Set x-axis ticks to integers
plt.savefig('../images/vrtest_value_sample2.png', dpi=300, bbox_inches='tight')


# Equal Weighted
# 1991-2006
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(equal_sample1['vr_stat'], label='VR Statistic',
        linewidth=1.5, marker='o')
# Confidence Interval
ax.fill_between(
    equal_sample1.index,
    equal_sample1['vr_stat']-cval*equal_sample1['vr_std_HLZ'],
    equal_sample1['vr_stat']+cval*equal_sample1['vr_std_HLZ'],
    color='skyblue', alpha=1, label='CI (HLZ)'
)
ax.fill_between(
    equal_sample1.index,
    equal_sample1['vr_stat']-cval*equal_sample1['vr_std_LM'],
    equal_sample1['vr_stat']+cval*equal_sample1['vr_std_LM'],
    color='skyblue', alpha=.4, label='CI (LM)'
)
# Baseline at 1
ax.axhline(1, color='black', linewidth=.5)
ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_xlabel('Aggregation Value (q)')
fig.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(1))  # Set x-axis ticks to integers
plt.savefig('../images/vrtest_equal_sample1.png', dpi=300, bbox_inches='tight')

# 2007-2022
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(equal_sample2['vr_stat'], label='VR Statistic',
        linewidth=1.5, marker='o')
# Confidence Interval
ax.fill_between(
    equal_sample2.index,
    equal_sample2['vr_stat']-cval*equal_sample2['vr_std_HLZ'],
    equal_sample2['vr_stat']+cval*equal_sample2['vr_std_HLZ'],
    color='skyblue', alpha=1, label='CI (HLZ)'
)
ax.fill_between(
    equal_sample2.index,
    equal_sample2['vr_stat']-cval*equal_sample2['vr_std_LM'],
    equal_sample2['vr_stat']+cval*equal_sample2['vr_std_LM'],
    color='skyblue', alpha=.4, label='CI (LM)'
)
# Baseline at 1
ax.axhline(1, color='black', linewidth=.5)
ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_xlabel('Aggregation Value (q)')
fig.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(1))  # Set x-axis ticks to integers
plt.savefig('../images/vrtest_equal_sample2.png', dpi=300, bbox_inches='tight')
