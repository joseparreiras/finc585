"""
FINC 585-3: Asset Pricing
Assignment 1 (Solutions)
Prof. Torben Andersen & Zhengyang Jiang
TA: Jose Antunes-Neto

This assignment is a solution for the Spearman Rank Correlation Test problem in the course. The question asks you to implement the LM Variance Ratio test using the S&P 500 data. The dataset is made available in the beginning of the course but it can also be found in the `data/` folder.
"""

# Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy as sp
import warnings

# Graphic Settings
sns.set_theme(context='notebook',style='whitegrid',palette='colorblind',font='sans-serif',font_scale=1.25)

# Importing the S&P 500 daily data
sp500 = pd.read_csv('../data/SP500_daily.csv', index_col=0)
sp500.index = pd.to_datetime(sp500.index, format='%Y%m%d')


def spearman_rank_test(x):
    """
    This function calculates the Spearman Rank Test for a set of returns. The test is based on Lecture 1 - Returns Predictability.

    Args:
        x (array): Series of returns

    Returns:
        dict: Dictionary containing the test statistics, its p-value, and the 95% confidence interval
    """
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
value_sample1 = spearman_rank_test(
    sp500['1991':'2006']['vwretd'])  # Value weighted
equal_sample1 = spearman_rank_test(
    sp500['1991':'2006']['ewretd'])  # Equal weighted

# (b) Second, do the same test for the sample period 2007-
value_sample2 = spearman_rank_test(
    sp500['2007':]['vwretd'])  # Value weighted
equal_sample2 = spearman_rank_test(
    sp500['2007':]['ewretd'])  # Equal weighted

# (c) Repeat the above, but using the absolute returns in lieu of the returns
# 1991-2006
absvalue_sample1 = spearman_rank_test(
    sp500['1991':'2006']['vwretd'].abs())  # Value weighted
absequal_sample1 = spearman_rank_test(
    sp500['1991':'2006']['ewretd'].abs())  # Equal weighted

absvalue_sample2 = spearman_rank_test(
    sp500['2007':]['vwretd'].abs())  # Value weighted
absequal_sample2 = spearman_rank_test(
    sp500['2007':]['ewretd'].abs())  # Equal weighted

# Print tables
value_table = pd.DataFrame(
    [value_sample1, value_sample2, absvalue_sample1, absvalue_sample2],
    index=['1991-2006', '2007-', '1991-2006 (Abs)', '2007- (Abs)']
)

equal_table = pd.DataFrame(
    [equal_sample1, equal_sample2, absequal_sample1, absequal_sample2],
    index=['1991-2006', '2007-', '1991-2006 (Abs)', '2007- (Abs)']
)

# Export tables to LaTeX
value_table.to_latex(
    '../tables/value_table.tex',
    float_format='%.3f',
    caption='Spearman Rank Correlation Test for Value Weighted S\&P 500 Returns',
    label='tab:value_table',
    header=['Z', 'pval', '{95% CI}', '{95% CI}'],
    position='!htbp',
    column_format='lS[table-format=-1.3]S[table-format=1.3]>{{[}}S[table-format=-2.3, table-space-text-pre={[}]@{;}S[table-format=-2.3, table-space-text-post={[]}]<{{]}}',
)

equal_table.to_latex(
    '../tables/equal_table.tex',
    float_format='%.3f',
    caption='Spearman Rank Correlation Test for Equal Weighted S\&P 500 Returns',
    label='tab:equal_table',
    header=['Z', 'pval', '{95% CI}', '{95% CI}'],
    position='!htbp',
    column_format='lS[table-format=-1.3]S[table-format=1.3]>{{[}}S[table-format=-2.3, table-space-text-pre={[}]@{;}S[table-format=-2.3, table-space-text-post={[]}]<{{]}}',
)