<center>
<h1> FINC 585-3 - Asset Pricing </h1>
<h1>💾 Data Description</h1>

<p>

*Prof: Torben Andersen & Zhengyang Jiang*

*TA: Jose Antunes-Neto*
</p>
</center>

These are the datasets used in the course. You will need to download them for the weekly assignments. If there is any issue with the data, contact me by [email](mailto:jose.neto@kellogg.northwestern.edu). Codes to update this dataset are available on the [GitHub repository](https://github.com/joseparreiras/finc585). The datasets are described below:

- [CRSP Daily Stock Data](#crsp-daily-stock-data)
- [Small Cap Daily Index](#small-cap-daily-index)
- [S\&P 500 Daily Index](#sp-500-daily-index)
- [S\&P 500 1-minute Price Data](#sp-500-1-minute-price-data)
- [Shiller CAPE Data](#shiller-cape-data)


## CRSP Daily Stock Data

The file [CRSP_daily.csv](CRSP_daily.csv) contains the return series of CRSP firms listed on the NYSE, AMEX, NASDAQ or ARCA at the daily frequency. These returns are calculated as percentage changes in the closing price of the stock. In the beginning of the sample, approximately 520 stocks are included. Towards the end of the sample approximately 7520 stocks are included. The data is downloaded from Wharton Research Data Services (WRDS) and is available from 1926-01-02 to 2023-12-29. The file contains the following variables:

| Variable | Type   | Description                                           |
| :------- | :----- | :---------------------------------------------------- |
| date     | string | Date of the observation in the format yyyymmdd        |
| vwretd   | float  | CRSP value-weighted index return (Dividends included) |
| ewretd   | float  | CRSP equal-weighted index return (Dividends included) |

More information can be found in the [CRSP website](https://www.crsp.org/products/documentation/stock-file-indexes-0). Data is contained in the [crsp_q_stock.dsi](https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_q_stock/msi/) table.

## Small Cap Daily Index

The file [smallcap_daily.csv](smallcap_daily.csv) contains the return series of the bottom 30% of CRSP firms listed on NYSE, AMEX or NASDAQ, ordered by size. Returns are calculated as a percentage change in the closing price of the index and are displayed at the percentage level. Data was obtained from [Kenneth French's website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) by downloading the [Portfolios Formed on Size Daily] [csv file](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_ME_CSV.zip). Data is available from 1926-07-01 to 2024-01-31. The file contains the following variables:

| Variable | Type   | Description                                                  |
| :------- | :----- | :----------------------------------------------------------- |
| DATE     | string | Date of the observation in the format yyyymmdd               |
| vw       | float  | Returns of the value-weighted portfolio (Dividends included) |
| ew       | float  | Returns of the equal-weighted portfolio (Dividends included) |

More information can be found [here](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_port_form_sz.html).

## S&P 500 Daily Index

The file [SP500_daily.csv](SP500_daily.csv) contains the returns of the S&P 500 index. Returns are calculated using the closing price of the index and are displayed at the percentage level. The data is downloaded from Wharton Research Data Services (WRDS) and is available from 1926-01-02 to 2023-12-29. The file contains the following variables:

| Variable | Type   | Description                                              |
| :------- | :----- | :------------------------------------------------------- |
| caldt    | string | Date of the observation in the format yyyymmdd           |
| vwretd   | float  | S&P 500 value-weighted index return (Dividends included) |
| ewretd   | float  | S&P 500 equal-weighted index return (Dividends included) |

Data is contained in the [crsp_q_indexes.dsp500](https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_q_indexes/msp500/) table.

## S&P 500 1-minute Price Data

The file [SPY_HF.csv](SPY_HF.csv) contains the price series of the S&P 500 ETF, SPY at the minute frequency. The data is download from NYSE Trades and Quotes (TAQ) [Consolidated Trades](https://wrds-www.wharton.upenn.edu/pages/get-data/nyse-trade-and-quote/millisecond-trade-and-quote-daily-product-2003-present-updated-daily/consolidated-trades/) database using WRDS's [SAS Studio platform](https://wrds-cloud.wharton.upenn.edu/SASStudio/). Observations are available from 1993-01-29 to 2023-02-27 from 9:30 to 16:00 (Eastern Time). The file contains the following variables:

| Variable | Type   | Description                                    |
| :------- | :----- | :--------------------------------------------- |
| DATE     | string | Date of the observation in the format yyyymmdd |
| TIME     | string | Time of the observation in the format HHMM     |
| PRICE    | float  | Close Price of the S&P 500 ETF at the minute   |

The TAQ database is divided into 2 types. For the series between 1993 and 2014, data is available at the second frequency and is obtained from the [taq](https://wrds-www.wharton.upenn.edu/pages/get-data/nyse-trade-and-quote/trade-and-quote-monthly-product-1993-2014/consolidated-trades/) library. For observations starting on 2015, data was collected at the milisecond level and is available in the [taqmsec](https://wrds-www.wharton.upenn.edu/pages/get-data/nyse-trade-and-quote/millisecond-trade-and-quote-daily-product-2003-present-updated-daily/consolidated-trades/) library. For both these series, the data was upscaled to the minute frequency using the last available price. More information about this dataset can be found at the NYSE [website](https://www.nyse.com/market-data/historical/daily-taq).

## Shiller CAPE Data

The file [Shiller_ie_data.csv](Shiller_ie_data.csv) data set consists of monthly stock price, dividends, and earnings data and the consumer price index (to allow conversion to real values), all starting January 1871. The data is directly downloaded from Robert Shiller's [website](https://www.econ.yale.edu/~shiller/data.htm) and more information can be found there.


# 🚀 Codes

* `getTAQ.sas`: SAS script example used in WRDS to download the S&P 500 1-minute price data; 
* `update_data.py`: Python script to update the remaining data. It downloads the data from the sources and saves it in the `data/` directory.

<!-- [^1]: SAS Studio is a web-based application that allows users to write and execute SAS code through a web browser. It is available to all Wharton students and faculty through the WRDS platform. It is commonly used when dealing with large datasets that require processing power beyond what is available in a personal computer and it tends to be faster than running the same code in a local machine in other languages such as Python or R. This should be a good time to learn more about as you will probably use it in the future. The code I used to extract the TAQ database is also available on the [GitHub Repository](https://github.com/joseparreiras/finc585) -->