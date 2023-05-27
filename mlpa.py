#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 16:08:29 2023

@author: bstaverosky
"""

import numpy as np
import pandas as pd
import sklearn
#import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
#import cvxpy as cp
import statsmodels.api as sm
import talib as talib


# Verify the versions of the loaded libraries (optional)
print("NumPy version:", np.__version__)
print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)
#print("matplotlib version:", plt.__version__)
print("seaborn version:", sns.__version__)
print("yfinance version:", yf.__version__)
#print("cvxpy version:", cp.__version__)
print("statsmodels version:", sm.__version__)

# Load macroeconomic variables from disk

infl = pd.read_csv("/home/bstaverosky/Documents/projects/MLPA/infl.csv", skiprows=3, header=1)
ey = pd.read_csv("/home/bstaverosky/Documents/projects/MLPA/earnings_yield.csv", skiprows=3, header=1)
dy = pd.read_csv("/home/bstaverosky/Documents/projects/MLPA/div_yield.csv", skiprows=3, header=1)
aaa = pd.read_csv("/home/bstaverosky/Documents/projects/MLPA/AAA_corporate_rate.csv", skiprows=3, header=1)
tr = pd.read_csv("/home/bstaverosky/Documents/projects/MLPA/20year_treasury_rate.csv", skiprows=3, header=1)
pb = pd.read_csv("/home/bstaverosky/Documents/projects/MLPA/price_to_book.csv", skiprows=3, header=1)
ts = pd.read_csv("/home/bstaverosky/Documents/projects/MLPA/term_spread.csv", skiprows=3, header=1)


# Load SPY
ticker = "SPY"
asset = yf.download(ticker, start='1900-01-01', progress=True)

# Calculate signals
# SMA RATIO
asset['sma_rat'] = np.log(talib.SMA(asset['Close'], timeperiod=21)/talib.SMA(asset['Close'], timeperiod = 252))
            
# VOL RATIO
for i in range(len(asset.index)):
    asset.loc[asset.index[i], "stvol"] = np.std(np.diff(np.log(asset.loc[asset.index[1:i], "Close"].tail(65))))
    asset.loc[asset.index[i], "ltvol"] = np.std(np.diff(np.log(asset.loc[asset.index[1:i], "Close"].tail(252))))
    asset.loc[asset.index[i], "vol_rat"] = asset.loc[asset.index[i], "stvol"]/asset.loc[asset.index[i], "ltvol"]
                
# PRICE TO HIGH
for i in range(len(asset.index)):
    asset.loc[asset.index[i], "p2h"] = asset.loc[asset.index[i], "Close"]/np.max(asset.loc[asset.index[(i-252):(i-1)], "Close"])
    
    
    
test = 


