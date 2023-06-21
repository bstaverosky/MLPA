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
from sklearn.ensemble import GradientBoostingRegressor
import pyfolio as pf
from datetime import datetime
import uuid
import pickle


# Load Adaptive Leverage Model as Benchmark
adaptivelev = pd.read_csv("/home/bstaverosky/Documents/projects/adaptive_leverage/asset_output.csv")
adaptivelev.set_index('Index', inplace=True)
alev_strat = adaptivelev.loc[:, "strat"]
alev_model = adaptivelev.loc[:, ["score"]]

# Load SPY
ticker = "^GSPC"
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

# Load macroeconomic variables from disk
s
# Read the CSV files and set the "Date" column as the index while converting to daily frequency
data_files = [
    ("/home/bstaverosky/Documents/projects/MLPA/infl.csv", "infl"),
    ("/home/bstaverosky/Documents/projects/MLPA/earnings_yield.csv", "ey"),
    ("/home/bstaverosky/Documents/projects/MLPA/div_yield.csv", "dy"),
    ("/home/bstaverosky/Documents/projects/MLPA/AAA_corporate_rate.csv", "aaa"),
    ("/home/bstaverosky/Documents/projects/MLPA/20year_treasury_rate.csv", "tr"),
    ("/home/bstaverosky/Documents/projects/MLPA/price_to_book.csv", "pb"),
    ("/home/bstaverosky/Documents/projects/MLPA/term_spread.csv", "ts")
]

dfs_to_merge = []

#file_path = "/home/bstaverosky/Documents/projects/MLPA/infl.csv"

# Put only the relevent columns from each data frame in a row and rename them
for file_path, var_name in data_files:
    df = pd.read_csv(file_path, skiprows=3, header=1, index_col="Date")
    df = df.loc[df.index[0:len(df)],["Value"]]
    df.columns = [var_name]
    dfs_to_merge.append(df)
    
# Merge the DataFrames by index date
merged_df = asset.copy()

for df in dfs_to_merge:
    merged_df = merged_df.merge(df, left_index=True, right_index=True, how='outer')

merged_df = merged_df.fillna(method = "ffill")
#merged_df = merged_df.dropna()

asset = merged_df

##### PARAMETERS #####
expanding_window = False

algos = ["xg"]
numest = [5]
nest = 10
maxdepth = [5]
mdepth = 5
#prediction window
predwindow = [21]
#pw = 10
#lookback window
lw = 2560
# ticker
#ticker = "SPY"
alg = "xg"


#for pw in predwindow:
#    for nest in numest:
#        for mdepth in maxdepth:
    
            backtest_id = str(uuid.uuid4())
            
            pw = 21
            lw = 2560
            
            # Get Daily Return
            asset['dayret'] = asset['Close'].pct_change()
            
            # Get Weekly Return
            #test = asset.resample('W').ffill()
            asset['closelag'] = asset['Close'].shift(pw)
            def percentage_change(col1,col2):
                return ((col2 - col1) / col1) * 100
            
            asset['pwret'] = percentage_change(asset['closelag'],asset['Close'])
            # Shift so that your prediction period aligns with the current day plus
            # one additional day to fix look ahead bias
            asset['pwret'] = asset['pwret'].shift(-(pw+1))
    
            if expanding_window = True:
                
    
            # CLEAN DATAFRAME
            #df = asset[['sma_rat', 'vol_rat', 'p2h', 'infl', 'ey', 'dy', 'aaa', 'tr', 'pb','ts', 'pwret']].tail(len(asset)-lw)
            df = asset[['sma_rat', 'vol_rat', 'p2h', 'infl', 'ey', 'dy', 'ts', 'pwret']].tail(len(asset)-lw)
            df = df.dropna()
            #features = ['sma_rat', 'vol_rat', 'p2h', 'infl', 'ey', 'dy', 'aaa', 'tr', 'pb','ts']
            features = ['sma_rat', 'vol_rat', 'p2h', 'infl', 'ey', 'dy', 'ts']
            
            predf = pd.DataFrame(columns = ["pred"])
            
            # Create rolling window trainset
            for i in range((lw+pw), len(df.index)-1):
                print(i)
                
                if alg == "xg":
                    model = GradientBoostingRegressor(n_estimators = nest,
                                                      max_depth=mdepth,
                                                      random_state=2)
                elif alg == "linreg":
                    print("linreg")
                    model = LinearRegression()
                
                # Make trainsets
                xtrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],features]
                ytrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['pwret']]
                    
                # Make testsets
                xtest = df.loc[[df.index[i+1]],features]    
                ytest = df.loc[[df.index[i+1]],['pwret']]
                type(ytest)
                    
                model.fit(xtrain, ytrain)
                y_pred = model.predict(xtest)
                    
                #results = pd.DataFrame(data = )
                    
                lframe = pd.DataFrame(y_pred, columns = ["pred"], index = ytest.index)
                predf = predf.append(lframe)
                

                
            # Put predictions back on original data frame
            # And convert y_pred so it can be added to dataframe
            # Put predictions back on original data frame
            # And convert y_pred so it can be added to dataframe
            
            sframe = df
            sframe['signal'] = predf
            sframe['signal'] = sframe['signal'].shift(1)
            sframe['return'] = asset['dayret']
            
            # Create the strategy return performance
            for i in range(len(sframe.index)):
                if sframe.loc[sframe.index[i], "signal"] > 0:
                    sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*0.75
                else:
                    sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*-0.10
                    
            bmk_series = sframe.loc[:,"return"].tail(len(sframe)-(lw+pw))
            strat_series = sframe.loc[:,"strat"].tail(len(sframe)-(lw+pw))
            
            #tsheet = pf.create_simple_tear_sheet(returns = strat_series, benchmark_rets=bmk_series)
            
            pf.create_simple_tear_sheet(returns = strat_series, benchmark_rets=bmk_series)
            
            # Real-time forward week prediction
            #rtpred = asset.loc[[asset.index[len(asset)-1]], ['sma_rat', 'vol_rat', 'p2h']]
            #model.predict(rtpred)
            sframe = sframe.dropna()
            # Evaluate the test set RMSE
            #MSE(sframe.loc[:,"pwret"], sframe.loc[:,"signal"])**(1/2)
            
            
            # Performance Data Frame
            perf = pd.DataFrame({
                'Date Run': datetime.today().strftime('%Y-%m-%d'),
                'Backtest_id':backtest_id
                'Ticker': ticker,
                'algo': alg,
                'Prediction Window': [pw],
                'Lookback Window': [lw],
                'Number of Estimators': nest,
                'Max Depth': mdepth,
                'Annualized Return': ep.cagr(strat_series),
                'Benchmark Annualized Return': ep.cagr(bmk_series),
                'Active Annualized Return': ep.cagr(strat_series)-ep.cagr(bmk_series),
                'Cumulative Returns': ep.cum_returns_final(strat_series)*100,
                'Sharpe Ratio': ep.sharpe_ratio(strat_series),
                'Benchmark Sharpe Ratio': ep.sharpe_ratio(bmk_series),
                'Sortino Ratio': ep.sortino_ratio(strat_series),
                'Max Drawdown': ep.max_drawdown(strat_series),
                'Mean Squared Error': MSE(sframe.loc[:,"pwret"], sframe.loc[:,"signal"])**(1/2)
                #'Baseline': (sframe.loc[:,"pwret_bin"] == 1).sum()/len(sframe),
                #'Accuracy': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"]),
                #'Skill': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"])-(sframe.loc[:,"pwret_bin"] == 1).sum()/len(sframe)
            })

            # Save Summary Statistics to CSV
            if path.exists('/home/brian/Documents/projects/MLPA/Summary' + "/" + "MLPA_backtest_summary.csv") == True:
                perf.to_csv('/home/brian/Documents/projects/shared_projects/Data' + "/" + "MLPA_backtest_summary.csv", mode = 'a', header = False)
            elif path.exists('/home/brian/Documents/projects/shared_projects/Data' + "/" + "MLPA_backtest_summary.csv") == False:
                perf.to_csv('/home/brian/Documents/projects/shared_projects/Data' + "/" + "MLPA_backtest_summary.csv", header = True)
             
            # Save last relevant model 
            with open("/home/bstaverosky/Documents/projects/MLPA/models/" + backtest_id + datetime.today().strftime('%Y-%m-%d') + "model.pkl", 'wb') as file:
                pickle.dump(model, file)
                
            # Save Backtest Time Series
            if path.exists('/home/brian/Documents/projects/MLPA/backtests/backtest_ts_' + backtest_id + ".csv") == True:
                strat_series.to_csv('/home/brian/Documents/projects/MLPA/backtests/backtest_ts_' + backtest_id + ".csv", mode = 'a', header = False)
            elif path.exists('/home/brian/Documents/projects/MLPA/backtests/backtest_ts_' + backtest_id + ".csv") == False:
                strat_series.to_csv('/home/brian/Documents/projects/MLPA/backtests/backtest_ts_' + backtest_id + ".csv", header = True)


    




    
    


