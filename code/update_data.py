"""
Author: Jose Antunes-Neto

This file is intended to updated the database with the latest data for the Asset Pricing course (FINC 585-3) at Kellogg School of Management, Northwestern University.
"""

import urllib
import json
import os
import wrds
import pandas as pd
import datetime as dt
import numpy as np
from tqdm import tqdm
import scipy.io as sio

import json

# Load WRDS Credentials
with open('config.json', 'r') as file:
    cred = json.load(file)

# Use WRDS API to download the data
db = wrds.Connection(wrds_username=cred['username'])

# CRSP -------------------------------------------------------------------------
"""
This dataset contains daily stock returns and market capitalization for all NYSE, AMEX, and NASDAQ stocks. It is imported form the CRSP database available on Wharton Research Data Services (WRDS). The corresponding WRDS table is crsp_q_stock.dsi
"""

crsp = db.get_table("crsp_q_stock", "dsi")
# Subset the data to obtain only the value weighted and equal weighted returns
crsp = crsp[["date", "vwretd", "ewretd"]]
crsp.dropna(inplace=True)  # Drop first NA row
# Rename first column to DATE
crsp.rename({"date": "DATE"}, axis=1, inplace=True)
# Reformat the date to %Y%m%d
crsp["DATE"] = pd.to_datetime(crsp["DATE"], format="%Y-%m-%d")
crsp["DATE"] = crsp["DATE"].dt.strftime("%Y%m%d")
# Overwrite the old file
crsp.to_csv("data/CRSP_daily.csv", index=False)

# Fama French Small Cap --------------------------------------------------------
"""
This dataset contains return series of the bottom 30% of CRSP firms listed on NYSE, AMEX or NASDAQ. It is obtained from Kenneth French's website on Portfolios Formed on Size [Daily] [CSV].
"""

# URL for the data
url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_ME_Daily_CSV.zip"
# Read the data
size_port = pd.read_csv(url, skiprows=12, compression="zip", header=0, index_col=0)
# size_port.rename({'Unnamed: 0':'DATE'}, axis = 1, inplace = True)
# Find the divider between the two series
divider = np.where(size_port.index.isna())[0][0]

# Value Weighted
# Subset the value weighted series
vw_smallcap = size_port.iloc[: divider - 1, 1].rename("vw").astype(float)
# Convert the index to datetime
vw_smallcap.index = pd.to_datetime(vw_smallcap.index, format="%Y%m%d")
# vw_smallcap.index = vw_smallcap['DATE']

# Equaly Weighted
# Subset the equal weighted series
ew_smallcap = size_port.iloc[divider + 1 :, 1].rename("ew").astype(float)
# Convert the index to datetime
ew_smallcap.index = pd.to_datetime(ew_smallcap.index, format="%Y%m%d")

# Merge
smallcap = pd.DataFrame({"vw": vw_smallcap, "ew": ew_smallcap}).reset_index()
smallcap.rename({"index": "DATE"}, axis=1, inplace=True)  # Rename date column
smallcap["DATE"] = smallcap["DATE"].dt.strftime("%Y%m%d")  # Reformat the date to %Y%m%d
# Overwrite the old file
smallcap.to_csv("data/smallcap_daily.csv", index=False)

# S&P Daily Returns ------------------------------------------------------------
"""
This dataset contains the returns of the S&P 500 index. It is obtained from the CRSP database available on WRDS.
"""

sp500 = db.get_table("crsp_q_indexes", "dsp500")  # Get the data
# Select the value/equal weighted returns
sp500 = sp500[["caldt", "vwretd", "ewretd"]]
sp500.dropna(inplace=True)  # Drop first NA row
sp500["caldt"] = pd.to_datetime(sp500["caldt"], format="%Y-%m-%d").dt.strftime(
    "%Y%m%d"
)  # Reformat the date to %Y%m%d
# Overwrite the old file
sp500.to_csv("data/SP500_daily.csv", index=False)

# Shiller CAPE -----------------------------------------------------------------

url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"  # URL of the data
outfile = "data/Shiller_ie_Data.xls"  # File to save the data
urllib.request.urlretrieve(url, outfile)  # Request download from URL

db.close()  # Close the connection to WRDS

# Fama French 3 Daily Factors --------------------------------------------------

# Obtain FF5 factors
ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip" # Url
ffdaily = pd.read_csv(ff5_url, compression = "zip", skiprows = 3, skipfooter = 2) # Download 
# Rename columns 
ffdaily.rename({"Unnamed: 0": "date"}, axis = 1, inplace = True)
ffdaily.rename({x: x.lower() for x in ffdaily.columns}, axis = 1, inplace = True)
# Format date
ffdaily['date'] = pd.to_datetime(ffdaily['date'], format = '%Y%m%d')

# Obatin momentum factor
mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip" # Url
# Download
mom = pd.read_csv(mom_url, compression = "zip", skiprows = 13, skipfooter = 2)
# Rename columns
mom.rename({"Unnamed: 0": "date"}, axis = 1, inplace = True)
mom.rename({x: x.lower().strip() for x in mom.columns}, axis = 1, inplace = True)
# Format date
mom['date'] = pd.to_datetime(mom['date'], format = '%Y%m%d')

# Merge the two datasets
ffdaily = ffdaily.merge(mom, how = 'left', on = 'date')
ffdaily.set_index('date', drop = True, inplace = True)
ffdaily = np.log(1+ffdaily/100)
# Export
ffdaily['2004':].to_csv("../data/ffdaily.csv", index = True, header = True)