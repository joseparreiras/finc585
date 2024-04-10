"""
FINC 585-3: Asset Pricing
Assignment 2 (Solutions)
Prof. Torben Andersen
TA: Jose Antunes-Neto
"""

# Modules
import pandas as pd
import numpy as np
from fredapi import Fred
import statsmodels.api as sm
from stargazer import Stargazer

# Problem 6. -------------------------------------------------------------------
"""
This problem assess the predictability of returns using macro-finance variables. We will use S&P 500 excess returns as the market indicator and the set of macro-finance variables from Shiller (2000) as the set of predictors. 
"""

start = '1963-01-01'
end = '2022-12-31'

# Data
# S&P 500 daily returns
sp500 = pd.read_csv('../data/SP500_daily.csv',
                    index_col=0, parse_dates=True)
sp500 = sp500[start:end]  # Select the period
sp500 = (1+sp500).resample('M').prod()-1  # Upscale to monthly returns
# Set the date to the first day of the month for convention
sp500.index = sp500.index.map(lambda x: x.date().replace(day=1))

# 3 Month T-Bill
# Use FRED API to get data
fred = Fred(api_key='126cf1d787c6c0f347038abffe2f86d0')
tbill = fred.get_series('TB3MS')  # Get data
tbill = tbill/12/100  # Convert to monthly returns

# S&P 500 excess returns
sp500 = sp500.apply(lambda x: x-tbill.shift(1), axis=0).dropna()

# Shiller macro-finance variables
shiller = pd.read_excel('../data/Shiller_ie_data.xls',
                        usecols=['Date', 'P', 'D', 'E', 'CAPE'], sheet_name='Data', skiprows=7, skipfooter=1, dtype={'Date': str})
# Correct excel date format (October appears as YYYY.1, not YYYY.10)
shiller['Date'] = shiller['Date'].apply(
    lambda x: x if len(x) == 7 else x[:5] + '10')
# Format index to datetime
shiller.index = pd.to_datetime(shiller['Date'], format='%Y.%m')
shiller.drop('Date', axis=1, inplace=True)  # Drop the date column
shiller = shiller[start:end]  # Select the period

# Create dataframe of predictors
predictors = pd.DataFrame(index=shiller.index)
# Construct the P/E and D/P ratios
predictors['PE'] = shiller['P']/shiller['E']
predictors['DP'] = shiller['D']/shiller['P']
# Construct the relative interest rate
rrel = tbill-tbill.rolling(12).mean().shift(1)
predictors['RREL'] = rrel[start:end]
predictors = predictors.shift(1)  # Lagged predictors

# Regressions

value_models = dict()
equal_models = dict()

# X: P/E
value_models.update(
    {'a': sm.OLS(sp500['vwretd'], sm.add_constant(predictors['PE']), missing='drop')})
equal_models.update(
    {'a': sm.OLS(sp500['ewretd'], sm.add_constant(predictors['PE']), missing='drop')})

# X: D/P
value_models.update(
    {'b': sm.OLS(sp500['vwretd'], sm.add_constant(predictors['DP']), missing='drop')})
equal_models.update(
    {'b': sm.OLS(sp500['ewretd'], sm.add_constant(predictors['DP']), missing='drop')})

# X: RREL
value_models.update({'c': sm.OLS(
    sp500['vwretd'], sm.add_constant(predictors['RREL']), missing='drop')})
equal_models.update({'c': sm.OLS(
    sp500['ewretd'], sm.add_constant(predictors['RREL']), missing='drop')})

# X: P/E, RREL
value_models.update({'d': sm.OLS(sp500['vwretd'], sm.add_constant(
    predictors[['PE', 'RREL']]), missing='drop')})
equal_models.update({'d': sm.OLS(sp500['ewretd'], sm.add_constant(
    predictors[['PE', 'RREL']]), missing='drop')})

# Fit the models
# Value weighted S&P 500
value_reg = dict()
for key in value_models.keys():
    value_reg.update({key: value_models[key].fit()})

# Equal weighted S&P 500
equal_reg = dict()
for key in equal_models.keys():
    equal_reg.update({key: equal_models[key].fit()})

# White standard errors
# Value weighted S&P 500
value_reg_white = dict()
for key in value_models.keys():
    value_reg_white.update({key: value_models[key].fit(cov_type='HC0')})

# Equal weighted S&P 500
equal_reg_white = dict()
for key in equal_models.keys():
    equal_reg_white.update({key: equal_models[key].fit(cov_type='HC0')})

# HAC standard errors
# Value weighted S&P 500
max_lags = 12  # Maximum number of lags to use in the HAC estimator
value_reg_hac = dict()
for key in value_models.keys():
    value_reg_hac.update({key: value_models[key].fit(
        cov_type='HAC', cov_kwds={'maxlags': max_lags})})

# Equal weighted S&P 500
equal_reg_hac = dict()
for key in equal_models.keys():
    equal_reg_hac.update({key: equal_models[key].fit(
        cov_type='HAC', cov_kwds={'maxlags': max_lags})})

# Stargazer


def mytable(reg, reg_white, reg_hac, title=None, label=None):
    """
    Function to construct the desired latex table contained homoskedastic, White and HAC t-stat.

    Args:
        reg (dict): Homoskedastic regression results.
        reg_white (dict): White regression results.
        reg_hac (dict): HAC regression results.
        title (str, optional): Title for the table. Defaults to None.
        label (str, optional): Latex label for the table. Defaults to None.

    Returns:
        str: Latex source code with the table
    """
    table = Stargazer(reg.values())
    table.title(title)
    table.covariate_order(['const', 'PE', 'DP', 'RREL'])  # Order of covariates
    table.custom_columns(['(a)', '(b)', '(c)', '(d)'], [
        1, 1, 1, 1])  # Custom column names
    table.rename_covariates(
        {'const': 'Constant', 'PE': 'P/E', 'DP': 'D/P', 'RREL': 'RREL'})
    table.show_model_numbers(False)  # Remove (1), (2), etc.
    table.add_custom_notes(
        ['Stars w.r.t homoskedastic std. errors', '( ): Homoskedastic t-stat; [ ]: White t-stat; \{ \}: HAC t-stat'])  # Add notes for additional t-statistics
    table.show_t_statistics(True)  # Show t-statistics instead of std. errors

    # T-Statistics
    # Homoskedastic
    homosk_tval = homosk_tval = pd.DataFrame({
        key: value.tvalues for key, value in reg.items()
    }).loc[['const', 'PE', 'DP', 'RREL']]
    homosk_tval = homosk_tval.round(3)  # Round to 3 decimals
    # Add ( ) around t-statistics
    homosk_tval = homosk_tval.applymap(
        lambda x: '(%0.3f)' % x if not np.isnan(x) else '')
    # Combine the t-statistics into a single table line
    homosk_tval = homosk_tval.apply(
        lambda x: ' & ' + ' & '.join(x) + '\\\\', axis=1)

    # White
    white_tval = pd.DataFrame({
        key: value.tvalues for key, value in reg_white.items()
    }).loc[['const', 'PE', 'DP', 'RREL']]
    white_tval = white_tval.round(3)
    # Add [ ] around t-statistics
    white_tval = white_tval.applymap(
        lambda x: '[%.3f]' % x if not np.isnan(x) else '')
    # Combine the t-statistics into a single table line
    white_tval = white_tval.apply(
        lambda x: ' & ' + ' & '.join(x) + '\\\\', axis=1)

    # HAC
    hac_tval = pd.DataFrame({
        key: value.tvalues for key, value in reg_hac.items()
    }).loc[['const', 'PE', 'DP', 'RREL']]
    hac_tval = hac_tval.round(3)
    # Add { } around t-statistics
    hac_tval = hac_tval.applymap(
        lambda x: '\{%.3f\}' % x if not np.isnan(x) else '')
    # Combine the t-statistics into a single table line
    hac_tval = hac_tval.apply(lambda x: ' & ' + ' & '.join(x) + '\\\\', axis=1)

    # Split the latex source code into lines
    latex = table.render_latex().split('\n')
    latex[0] += '\\footnotesize'  # Make the table smaller
    if label:
        latex[1] += '\\label{%s}' % label  # Add label
    # Change the color of the table
    latex[1] += '\\begingroup \\color{nu purple}'
    latex[-2] += '\\endgroup'
    count = 0  # Set counter for lines to replace
    for i, line in enumerate(latex):
        if count < 4:
            if 't=' in line:
                # Check if line corresponds to t-stats
                new_line = '\n'.join(
                    (homosk_tval[count], white_tval[count], hac_tval[count])) + '\\\\'
                # Replace with new one
                latex[i] = new_line
                count += 1
    return '\n'.join(latex)


# Value Weighted
value_table = mytable(value_reg, value_reg_white, value_reg_hac,
                      title='Testing for Returns Predictability (Value Weighted)', label='tab:value_reg')
with open('../tables/shiller_value_reg.tex', 'w') as f:
    f.write(value_table)

equal_table = mytable(equal_reg, equal_reg_white, equal_reg_hac,
                      title='Testing for Returns Predictability (Equal Weighted)', label='tab:equal_reg')

with open('../tables/shiller_equal_reg.tex', 'w') as f:
    f.write(equal_table)
