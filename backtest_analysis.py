#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 16:53:02 2023

@author: bstaverosky
"""

    import numpy as np
    import matplotlib
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import sklearn
    #import matplotlib.pyplot as plt
    import seaborn as sns
    import yfinance as yf
    #import cvxpy as cp
    import statsmodels.api as sm
    import talib as talib
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error as MSE
    import pyfolio as pf
    from datetime import datetime
    import uuid
    import pickle
    import empyrical as ep
    from os import path
    

btdata = pd.read_csv("/home/bstaverosky/Documents/projects/MLPA/summary/MLPA_shiller_backtest_summary.csv")

# Filter by Date
btdata["Date Run"] = pd.to_datetime(btdata["Date Run"])
btdata = btdata[btdata["Date Run"]>="2023-09-16"]
btdata = btdata[btdata["algo"]=="xg"]

output = {'Outperformance_Ratio': [len(btdata[btdata["Active Annualized Return"]>0])/len(btdata)],
          "Average Outperformance": [np.mean(btdata['Active Annualized Return'])],
          "Max Outperformance": [np.max(btdata["Active Annualized Return"])],
          "Min Outperformance": [np.min(btdata["Active Annualized Return"])],
          "Median Sharpe Ratio": [np.median(btdata["Sharpe Ratio"])],
          "Median Benchmark Sharpe Ratio": [np.median(btdata['Benchmark Sharpe Ratio'])],
          "Median Max Drawdown": [np.median(btdata["Max Drawdown"])]
          }
output = pd.DataFrame(output)

output.to_csv("/home/bstaverosky/Documents/projects/MLPA/summary/summary_panda.csv")



￼
￼
￼

