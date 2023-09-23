#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 21:15:52 2023

@author: bstaverosky
"""
#features = ['sma_rat', 'vol_rat', 'p2h', 'infl', 'ey', 'dy', 'ts']

### INDICATORS TO SOURCE ###
# 1. Inflation
# 2. Earnings Yield
# 3. Dividend Yield
# 4. Treasury Spread


### SHILLER DATA ###
# Sourcing:
# 1. Inflation
# 2. Earnings Yield
# 3. Dividend Yield

from datetime import date
import urllib.request
import yfinance as yf
import pandas as pd
from dateutil.relativedelta import relativedelta

def download_file(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        print("File downloaded successfully!")
    except Exception as e:
        print("Error occurred while downloading the file:", e)

# Example usage
file_url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
save_location = "/home/bstaverosky/Documents/projects/MLPA/shiller_data.xls"

download_file(file_url, save_location)
shiller = pd.read_excel("/home/bstaverosky/Documents/projects/MLPA/shiller_data.xls", sheet_name="Data", skiprows = 7)
shiller = shiller.drop(columns=['Unnamed: 13', 'Unnamed: 15'])

cnames = ["Date",
          "S&P_Comp_P",
          "Dividend_D",
          "Earnings_E",
          "Consumer_Price_Index_CPI",
          "Date_Fraction",
          "Long_Interest_Rate_GS10",
          "Real_Price",
          "Real_Dividend",
          "Real_Total_Return_Price",
          "Real_Earnings",
          "Real_TR_Scaled_Earnings",
          "CAPE_one",
          "CAPE_two",
          "Excess_CAPE_Yield",
          "Monthly_Total_Bond_Returns",
          "Real_Total_Bond_Returns",
          "10_Year_Annualized_Stock_Real_Return",
          "10_Year_Annualized_Bond_Real_Return",
          "Real_10_Year_Excess_Annualized_Returns"]

shiller.columns = cnames

### YFINANCE ###
# Sourcing:
# 1. Treasury Spread

bonds = {"^TYX", "^TNX", "^IRX"}
start_date = "1900-01-01"
end_date = date.today()
outlist = {}

for b in bonds:
    data = yf.download(b, start=start_date, end=end_date)
    outlist[b] = data["Adj Close"]
byields = pd.concat(outlist, axis=1)

byields = byields.fillna(method = "ffill")

# Merge bond yields and Shiller data
def convert_to_last_day_of_month(date_str):
    year, month = map(int, date_str.split('.'))
    last_day = pd.to_datetime(f"{year}-{month:02}-01") + relativedelta(months=1, days=-1)
    return last_day.strftime('%Y-%m-%d')


### MERGE EVERYTHING INTO ONE DATA FRAME ### 

shiller = shiller.dropna(subset=['Date'])
shiller['Date'] = shiller['Date'].astype(str)

shiller['Date'] = shiller['Date'].apply(convert_to_last_day_of_month)
shiller['Date'] = pd.to_datetime(shiller['Date'])

shiller.set_index("Date", inplace=True)

mdata = shiller.merge(byields, left_index = True, right_index=True, how="left")
mdata = mdata.fillna(method = "ffill")

mdata["TMS"] = mdata['^TYX'] - mdata['^TNX']
mdata['TMS2'] = mdata['^TYX'] - mdata['^IRX']

mdata.to_pickle("/home/bstaverosky/Documents/projects/MLPA/shiller_factors.pkl")

