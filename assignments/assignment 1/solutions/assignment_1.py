"""
FINC 585-3: Asset Pricing
Assignment 1 (Solutions)
Prof. Torben Andersen
TA: Jose Antunes-Neto
"""

# Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy as sp

# Problem 5. -------------------------------------------------------------------
"""
In this problem, we will apply the Spearman Rank Correlation Test to the S&P 500 data
"""

sp500 = pd.read_csv('data/SP500_daily.csv', index_col=0, parse_dates=True)

"""
I will start by constructing a function that calculates the test based on page 4 of Lecture 1 - Returns Predictability
"""


def spearman_rank_test(x):
    from scipy.stats import norm
    # Define two groups based on index
    odd_group = x[::2]
    even_group = x[1::2]
    # Equalize the group number
    n = min(len(odd_group), len(even_group))
    odd_group = odd_group[:n]
    even_group = even_group[:n]
    # Define the ranks
    odd_rank = odd_group.rank()
    even_rank = even_group.rank()
    # Construct the variables
    d = odd_rank.values-even_rank.values  # Rank difference
    R = (d**2).sum()  # Sum of squared rank difference
    Z = (6*R-n*(n**2-1))/(n*(n+1)*(n-1)**0.5)  # Spearman Statistic
    p_value = 2*(1-norm.cdf(np.abs(Z)))  # P-Value
    cval = norm.ppf(0.95)  # Critical value
    ci = [Z-cval, Z+cval]  # Confidence interval
    return {
        "Z": Z,
        "p_value": p_value,
        "ci_95_lo": ci[0],
        "ci_95_hi": ci[1]
    }


# (a) First, do the test as suggested for January 1991-December 2006 period
value9106 = spearman_rank_test(
    sp500['1991':'2006']['vwretd'])  # Value weighted
equal9106 = spearman_rank_test(
    sp500['1991':'2006']['ewretd'])  # Equal weighted

# (b) Second, do the same test for the sample period 2007-2022
value0722 = spearman_rank_test(
    sp500['2007':'2022']['vwretd'])  # Value weighted
equal0722 = spearman_rank_test(
    sp500['2007':'2022']['ewretd'])  # Equal weighted

# (c) Repeat the above, but using the absolute returns in lieu of the returns
# 1991-2006
absvalue9106 = spearman_rank_test(
    sp500['1991':'2006']['vwretd'].abs())  # Value weighted
absequal9106 = spearman_rank_test(
    sp500['1991':'2006']['ewretd'].abs())  # Equal weighted

absvalue0722 = spearman_rank_test(
    sp500['2007':'2022']['vwretd'].abs())  # Value weighted
absequal0722 = spearman_rank_test(
    sp500['2007':'2022']['ewretd'].abs())  # Equal weighted

# Print tables
value_table = pd.DataFrame(
    [value9106, value0722, absvalue9106, absvalue0722],
    index=['1991-2006', '2007-2022', '1991-2006 (Abs)', '2007-2022 (Abs)']
)

equal_table = pd.DataFrame(
    [equal9106, equal0722, absequal9106, absequal0722],
    index=['1991-2006', '2007-2022', '1991-2006 (Abs)', '2007-2022 (Abs)']
)

# Export tables to LaTeX
# value_table.to_latex(
#     'tables/value_table.tex',
#     float_format='%.3f',
#     caption='Spearman Rank Correlation Test for Value Weighted S\&P 500 Returns',
#     label='tab:value_table',
#     header=['{Z}', '{p-value}', '{95\% CI}', '{95\% CI}'],
#     position='!htbp',
#     column_format='lS[table-format=-1.3]S[table-format=1.3]>{{[}}S[table-format=-2.3, table-space-text-pre={[}]@{;}S[table-format=-2.3, table-space-text-post={[]}]<{{]}}',
# )

# equal_table.to_latex(
#     'tables/equal_table.tex',
#     float_format='%.3f',
#     caption='Spearman Rank Correlation Test for Equal Weighted S\&P 500 Returns',
#     label='tab:equal_table',
#     header=['{Z}', '{p-value}', '{95\% CI}', '{95\% CI}'],
#     position='!htbp',
#     column_format='lS[table-format=-1.3]S[table-format=1.3]>{{[}}S[table-format=-2.3, table-space-text-pre={[}]@{;}S[table-format=-2.3, table-space-text-post={[]}]<{{]}}',
# )

# Problem 6. -------------------------------------------------------------------
"""
In this problem we will apply the Variance Ratio test of Lo & MacKinlay (1988) to the CRSP data
"""

# Import CRSP daily data
crsp = pd.read_csv('data/CRSP_daily.csv', index_col=0, parse_dates=True)


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
            return 1/T*nd.sum(x*x.shift(j)*x*x.shift(k))
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
q_max = 20
cval = sp.stats.norm.ppf(0.95)


def vr_table(x, q_max=20):
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


value_9106 = vr_table(crsp['vwretd']['1991': '2006'], q_max=20)
value_0722 = vr_table(crsp['vwretd']['2007': '2022'], q_max=20)
equal_9106 = vr_table(crsp['ewretd']['1991': '2006'], q_max=20)
equal_0722 = vr_table(crsp['ewretd']['2007': '2022'], q_max=20)

# Graph
# Value Weighted
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# 1991-2006
ax.plot(value_9106['vr_stat'], label='VR Statistic',
        linewidth=1.5, color='blue', marker='o')
# Confidence Interval
ax.fill_between(
    value_9106.index,
    value_9106['vr_stat']-cval*value_9106['vr_std_HLZ'],
    value_9106['vr_stat']+cval*value_9106['vr_std_HLZ'],
    color='blue', alpha=0.2, label='CI (HLZ)'
)
ax.fill_between(
    value_9106.index,
    value_9106['vr_stat']-cval*value_9106['vr_std_LM'],
    value_9106['vr_stat']+cval*value_9106['vr_std_LM'],
    color='red', alpha=0.2, label='CI (LM)'
)
# Baseline at 1
ax.axhline(1, color='black', linewidth=.5)
ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_xlabel('Aggregation Value (q)')
fig.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(1))  # Set x-axis ticks to integers
plt.savefig('images/vrtest_value_9106.png', dpi=1200, bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# 2007-2022
ax.plot(value_0722['vr_stat'], label='VR Statistic',
        linewidth=1.5, color='blue', marker='o')
# Confidence Interval
ax.fill_between(
    value_0722.index,
    value_0722['vr_stat']-cval*value_0722['vr_std_HLZ'],
    value_0722['vr_stat']+cval*value_0722['vr_std_HLZ'],
    color='blue', alpha=0.2, label='CI (HLZ)'
)
ax.fill_between(
    value_0722.index,
    value_0722['vr_stat']-cval*value_0722['vr_std_LM'],
    value_0722['vr_stat']+cval*value_0722['vr_std_LM'],
    color='red', alpha=0.2, label='CI (LM)'
)
# Baseline at 1
ax.axhline(1, color='black', linewidth=.5)
ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_xlabel('Aggregation Value (q)')
fig.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(1))  # Set x-axis ticks to integers
plt.savefig('images/vrtest_value_0722.png', dpi=1200, bbox_inches='tight')


# Equal Weighted
# 1991-2006
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(equal_9106['vr_stat'], label='VR Statistic',
        linewidth=1.5, color='blue', marker='o')
# Confidence Interval
ax.fill_between(
    equal_9106.index,
    equal_9106['vr_stat']-cval*equal_9106['vr_std_HLZ'],
    equal_9106['vr_stat']+cval*equal_9106['vr_std_HLZ'],
    color='blue', alpha=0.2, label='CI (HLZ)'
)
ax.fill_between(
    equal_9106.index,
    equal_9106['vr_stat']-cval*equal_9106['vr_std_LM'],
    equal_9106['vr_stat']+cval*equal_9106['vr_std_LM'],
    color='red', alpha=0.2, label='CI (LM)'
)
# Baseline at 1
ax.axhline(1, color='black', linewidth=.5)
ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_xlabel('Aggregation Value (q)')
fig.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(1))  # Set x-axis ticks to integers
plt.savefig('images/vrtest_equal_9106.png', dpi=1200, bbox_inches='tight')

# 2007-2022
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(equal_0722['vr_stat'], label='VR Statistic',
        linewidth=1.5, color='blue', marker='o')
# Confidence Interval
ax.fill_between(
    equal_0722.index,
    equal_0722['vr_stat']-cval*equal_0722['vr_std_HLZ'],
    equal_0722['vr_stat']+cval*equal_0722['vr_std_HLZ'],
    color='blue', alpha=0.2, label='CI (HLZ)'
)
ax.fill_between(
    equal_0722.index,
    equal_0722['vr_stat']-cval*equal_0722['vr_std_LM'],
    equal_0722['vr_stat']+cval*equal_0722['vr_std_LM'],
    color='red', alpha=0.2, label='CI (LM)'
)
# Baseline at 1
ax.axhline(1, color='black', linewidth=.5)
ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_xlabel('Aggregation Value (q)')
fig.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(1))  # Set x-axis ticks to integers
plt.savefig('images/vrtest_equal_0722.png', dpi=1200, bbox_inches='tight')
