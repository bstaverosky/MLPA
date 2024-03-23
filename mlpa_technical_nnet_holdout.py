#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:07:32 2024

@author: brian
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import pyfolio as pf
from datetime import datetime
import uuid
import pickle
import empyrical as ep
from os import path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### DEFINE FUNCTIONS ###
def find_next_parameters_nnet(csv_path, algos, predwindow, lookbacks, neurons, epochs, activation_function):
    
    # Read the CSV File
    backtests = pd.read_csv(csv_path)
        
    # Sort the data in the order of execution
    backtests.sort_values(by=['Prediction Window', 'Lookback Window', 'algo', 'Neurons', 'Epochs', 'Activation Function'], inplace=True)
    
    # Sort the data in the order of execution
    #backtests.sort_values(by=['Prediction Window', 'Number of Estimators', 'Max Depth', 'Lookback Window', 'algo'], inplace=True)
    
    # Find the last row
    if not backtests.empty:
        last_run = backtests.iloc[-1]

    # Retrieve the parameters of the last run
        last_pw = last_run['Prediction Window']
        last_lw = last_run['Lookback Window']
        last_alg = last_run['algo']
        last_neu = last_run['Neurons']
        last_epo = last_run['Epochs']
        last_af = last_run['Activation Function']
        
        
    else:
        # If the CSV is empty, start from the beginning
        return predwindow[0], lookbacks[0], algos[0], neurons[0], epochs[0], activation_function[0]
    
    # Find the next set of parameters
    for pw in predwindow:
        for lw in lookbacks:
            for alg in algos:
                for neu in neurons:
                    for epo in epochs:
                        for af in activation_function:
                            if (pw, lw, alg, neu, epo, af) > (last_pw, last_lw, last_alg, last_neu, last_epo, last_af):
                                return pw, lw, alg, neu, epo, af
                        
    # if all combinations have been run, return None
    return None

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

# Get Daily Return
asset['dayret'] = asset['Close'].pct_change()

##### PARAMETERS #####
algos = ["nnet"]
#numest = [2,3,5,10]
#maxdepth = [1,2,3,5]
predwindow = [3,5,10,21,63,126,256]
lookbacks = [256, 512, 768, 1280, 1920, 2560, 3840]
epochs = [10,30,50,100]
neurons = [5,10,15,20,30,50]
activation_functions = ['relu', 'elu', 'selu']

### CREATE CSV PATH IF NECESSARY. READ IN IF EXISTING. ###
csv_path = "/home/brian/Documents/projects/MLPA/summary/MLPA_technical_nnet_holdout_summary.csv"

if os.path.exists(csv_path):
    next_params = find_next_parameters_nnet(csv_path, algos, predwindow, lookbacks, epochs, neurons, activation_functions)
else:
    column_names = ['Date Run',
                    'Backtest_id',
                    'Ticker',
                    'algo',
                    'Prediction Window',
                    'Lookback Window',
                    'Number of Estimators',
                    'Max Depth',
                    'Neurons',
                    'Epochs',
                    'Activation Function',
                    'Annualized Return',
                    'Benchmark Annualized Return',
                    'Active Annualized Return',
                    'Cumulative Returns',
                    'Sharpe Ratio',
                    'Benchmark Sharpe Ratio',
                    'Sortino Ratio',
                    'Max Drawdown',
                    'Mean Squared Error',
                    'Accuracy',
                    'Baseline Accuracy',
                    'Accuracy Model Improvement',
                    'Precision',
                    'Recall',
                    'F1 Score',
                    'Baseline F1 Score',
                    'F1 Score Model Improvement',
                    'Up Market Accuracy',
                    'Down Market Accuracy',
                    'Top 20% Up Market Accuracy',
                    'Bottom 20% Down Market Accuracy']
    df = pd.DataFrame(columns=column_names)
    df.to_csv(csv_path, index = False)

### READ IN BACKTEST PARAMETERS ALREADY RUN ###
backtests = pd.read_csv(csv_path)
# alg = "nnet"
# pw = 3
# lw = 256
# epo = 10
# neu = 5
# af = 'relu'
while next_params:
    pw, lw, alg, epo, neu, af = next_params
    
    print(pw)
    print(lw)
    print(alg)
    print(epo)
    print(neu)
    print(af)
    
    backtest_id = str(uuid.uuid4())
                            
    # Get Weekly Return
    asset['closelag'] = asset['Close'].shift(pw)
    def percentage_change(col1,col2):
        return ((col2 - col1) / col1) * 100
    
    ### LAG THE PREDICTION WINDOW ###
    asset['pwret'] = percentage_change(asset['closelag'],asset['Close'])
    # Shift so that your prediction period aligns with the current day plus
    # one additional day to fix look ahead bias
    asset['pwret'] = asset['pwret'].shift(-(pw+1))

    # CLEAN DATAFRAME
    #df = asset[['sma_rat', 'vol_rat', 'p2h', 'infl', 'ey', 'dy', 'aaa', 'tr', 'pb','ts', 'pwret']].tail(len(asset)-lw)
    
    columns_to_exclude = ['Open','High','Low','Close','Adj Close','Volume', 'closelag']
    
    df = asset.drop(columns = columns_to_exclude)
    
    df = df.dropna()
    
    ### Technical Features Only ###
    features = ['sma_rat', 'stvol', 'ltvol', 'vol_rat', 'p2h']
    
    predf = pd.DataFrame(columns = ["pred"])
    
    fi_list = []
    
    # Holdout train and test sets
    #for i in range((lw+pw), len(df.index)-1):

    # Create a sequential model
    model = Sequential()
    
    # Add an input layer and hidden layer with 10 neurons
    model.add(Dense(neu, input_shape=(5,), activation=af))
    
    #Additional hidden layers
    #model.add(Dense(20, activation='relu'))  # Second hidden layer with 20 neurons
    #model.add(Dense(15, activation='relu'))  # Third hidden layer with 15 neurons
    
    
    # Add a 1-neuron output layer
    model.add(Dense(1))
    
    
    # Compile your model
    model.compile(optimizer = 'adam', loss = 'mse', run_eagerly=True)
    
    # Train the model
    print("Training the neural net...")
                
    # Make trainsets
    xtrain = df.loc[df.index[1:round(len(df)/2)], features]
    ytrain = df.loc[df.index[1:round(len(df)/2)], ['pwret']]
    
    # Make testsets
    xtest = df.loc[df.index[(round(len(df)/2)+1):len(df)], features]
    ytest = df.loc[df.index[(round(len(df)/2)+1):len(df)], ['pwret']]
    
    type(ytest)
    
    model.fit(xtrain, ytrain, epochs = epo)

        
    y_pred = model.predict(xtest)
    y_pred = y_pred.flatten()
    
    ### WRITE PREDICTION TO DISK FOR RECALL ###
    
    predictions = {'Date': xtest.index.date,
                   'prediction': y_pred}
    
    predictions = pd.DataFrame(predictions)
    
    # fi_list.append(pd.DataFrame({
    #     'Feature': features,
    #     'Importance': model.feature_importances_
    # }))
        
    lframe = pd.DataFrame(y_pred, columns = ["pred"], index = ytest.index)
    predf = predf.append(lframe)
            
    # Put predictions back on original data frame
    # And convert y_pred so it can be added to dataframe
    # Put predictions back on original data frame
    # And convert y_pred so it can be added to dataframe
    
    ##### START FEATURE IMPORTANCE SECTION #####
    
    # combined_df = pd.concat(fi_list, ignore_index=True)
    # combined_df = pd.concat(fi_list, axis=1)
    # combined_df2 = pd.concat([fi_list[0]] + [df.iloc[:, 1] for df in fi_list[1:]], axis=1)
    
    # test = combined_df2.T
    
    # # Set the first row as the header
    # test.columns = test.iloc[0]

    # # Drop the first row as it's now the header
    # test = test.drop(test.index[0])
    
    # # Optionally, reset the index if you want to have a continuous integer index
    # test = test.reset_index(drop=True)

    # # Plot using the plot method
    # ax = test.plot(kind='line', title='Yearly Sales', grid=True, legend=True)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()

    # # Display the plot
    # plt.show()  
    
    # # Compute the mean for all columns
    # column_means = test.mean()

    # # Sort the means in descending order
    # sorted_means = column_means.sort_values(ascending=False)
    
    ##### END FEATURE IMPORTANCE SECTION #####
    
    sframe = df
    predf = predf[~predf.index.duplicated()]
    sframe['signal'] = predf
    sframe['signal'] = sframe['signal'].shift(1)

    sframe['return'] = sframe['dayret']
    sframe = sframe.dropna()
    sframe = sframe[~sframe.index.duplicated(keep='first')]
    
    pd.to_pickle(sframe, "/home/brian/Documents/projects/MLPA/sframe.pkl")
    sframe = pd.read_pickle("/home/brian/Documents/projects/MLPA/sframe.pkl")
    
    # Create categorical variables for when there are up/down days and 
    # for when the model predicts an up or down day.
    
    sframe['pwret_bin'] = (sframe['pwret'] > 0).astype(int)
    sframe['signal_bin'] = (sframe['signal'] > 0).astype(int)
    
    # Create the strategy return performance
    for i in range(len(sframe.index)):
        print(i)
        if sframe.loc[sframe.index[i], "signal"] > 0:
            sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*1.25
        else:
            sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*0.75
            
    bmk_series = sframe.loc[:,"return"].tail(len(sframe)-(lw+pw))
    strat_series = sframe.loc[:,"strat"].tail(len(sframe)-(lw+pw))
    
    #tsheet = pf.create_simple_tear_sheet(returns = strat_series, benchmark_rets=bmk_series)

    #pf.create_simple_tear_sheet(returns = strat_series, benchmark_rets=bmk_series)

    # Real-time forward week prediction
    #rtpred = asset.loc[[asset.index[len(asset)-1]], ['sma_rat', 'vol_rat', 'p2h']]
    #model.predict(rtpred)
    sframe = sframe.dropna()
    # Evaluate the test set RMSE
    #MSE(sframe.loc[:,"pwret"], sframe.loc[:,"signal"])**(1/2)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"])
    # Confusion matrix elements
    tn, fp, fn, tp = cm.ravel()
    # Create a baseline model 
    sframe[["baseline"]] = 1
    
    # Extreme Periods Analysis:
    # Mark the periods where the market displays tail returns
    # Measure how the model does in these periods
    
    up_days = sframe[sframe["dayret"]>0]
    down_days = sframe[sframe["dayret"]<0]
    
    # Calculate Rolling Quantiles for Each
    up_days['rolling_80th'] = up_days['dayret'].rolling(window=256, min_periods=1).quantile(0.8)
    down_days['rolling_20th'] = down_days['dayret'].rolling(window=256, min_periods=1).quantile(0.2)
    
    # Flag Returns
    up_days['flag'] = (up_days['dayret']>= up_days['rolling_80th']).astype(int)
    down_days['flag'] = (down_days['dayret'] <= down_days['rolling_20th']).astype(int)
    
    # Combine the Flags back into the original dataframe
    sframe['up_flag'] = 0 # Initialize the flag column
    sframe['down_flag'] = 0
    sframe.loc[up_days.index, 'up_flag'] = up_days['flag']
    sframe.loc[down_days.index, 'down_flag'] = down_days['flag']
    
    # Significant up days
    sig_up_days = up_days[up_days['flag']==1].loc[:,["signal_bin","flag"]]
    sig_down_days = down_days[down_days['flag']==1].loc[:,["signal_bin", "flag"]]                            
    
    perf = pd.DataFrame({
        'Date Run': datetime.today().strftime('%Y-%m-%d'),
        'Backtest_id':backtest_id,
        'Ticker': ticker,
        'algo': alg,
        'Prediction Window': pw,
        'Lookback Window': lw,
        'Number of Estimators': [np.nan],
        'Max Depth': [np.nan],
        'Neurons': neu,
        'Epochs': epo,
        'Activation Function': af,
        'Annualized Return': ep.cagr(strat_series),
        'Benchmark Annualized Return': ep.cagr(bmk_series),
        'Active Annualized Return': ep.cagr(strat_series)-ep.cagr(bmk_series),
        'Cumulative Returns': ep.cum_returns_final(strat_series)*100,
        'Sharpe Ratio': ep.sharpe_ratio(strat_series),
        'Benchmark Sharpe Ratio': ep.sharpe_ratio(bmk_series),
        'Sortino Ratio': ep.sortino_ratio(strat_series),
        'Max Drawdown': ep.max_drawdown(strat_series),
        'Mean Squared Error': MSE(sframe.loc[:,"pwret"], sframe.loc[:,"signal"])**(1/2),
        'Accuracy': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"]),
        'Baseline Accuracy': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"baseline"]),
        'Accuracy Model Improvement': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"]) - accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"baseline"]),
        'Precision': precision_score(sframe.loc[:,"pwret_bin"], sframe.loc[:, "signal_bin"]),
        'recall': recall_score(sframe.loc[:, "pwret_bin"], sframe.loc[:,"signal_bin"]),
        'F1 Score': f1_score(sframe.loc[:, "pwret_bin"], sframe.loc[:, "signal_bin"]),
        'Baseline F1 Score': f1_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"baseline"]),
        'F1 Score Model Improvement': f1_score(sframe.loc[:, "pwret_bin"], sframe.loc[:, "signal_bin"]) - f1_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"baseline"]),
        'Up Market Accuracy': tp / (tp + fn),
        'Down Market Accuracy': tn / (tn + fp),
        'Top 20% Up Market Accuracy': sig_up_days['signal_bin'].sum()/len(sig_up_days),
        'Bottom 20% Down Market Accuracy': (sig_down_days['signal_bin']==0).sum()/len(sig_down_days)    
    })

    # Save Summary Statistics to CSV
    # if path.exists('/home/brian/Documents/projects/MLPA/summary' + "/" + "MLPA_technical_backtest_summary.csv") == True:
    #     perf.to_csv('/home/brian/Documents/projects/MLPA/summary' + "/" + "MLPA_technical_backtest_summary.csv", mode = 'a', header = False, index = False)
    # elif path.exists('/home/brian/Documents/projects/MLPA/summary' + "/" + "MLPA_technical_backtest_summary.csv") == False:
    #     perf.to_csv('/home/brian/Documents/projects/MLPA/summary' + "/" + "MLPA_technical_backtest_summary.csv", header = True, index = False)
        
    # Save Summary Statistics to CSV
    if path.exists(csv_path) == True:
        perf.to_csv(csv_path, mode = 'a', header = False, index = False)
    elif path.exists(csv_path) == False:
        perf.to_csv(csv_path, header = True, index = False)    
     
    # Save last relevant model 
    with open("/home/brian/Documents/projects/MLPA/models/" + backtest_id + datetime.today().strftime('%Y-%m-%d') + "model.pkl", 'wb') as file:
        pickle.dump(model, file)
        
    # Find the next parameters to run
    next_params = find_next_parameters_nnet(csv_path, algos, predwindow, lookbacks, neurons, epochs, activation_functions)
                            
if not next_params:
    print("All combinations have been run.") 


