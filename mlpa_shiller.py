    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Sat May 27 16:08:29 2023
    
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
    
    # Load Shiller Data 
    shiller = pd.read_pickle("/home/bstaverosky/Documents/projects/MLPA/shiller_factors.pkl")
    
    
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
    
    ### COMBINE SHILLER FACTORS AND ADAPTIVE LEVERAGE SIGNALS ###
    
    merged_df = shiller.merge(asset, left_index=True, right_index=True, how = "right")
    merged_df = merged_df.fillna(method = "ffill")
    merged_df = merged_df.dropna()
    
    asset = merged_df
    
    # Get Daily Return
    asset['dayret'] = asset['Close'].pct_change()
    
    ##### PARAMETERS #####
    #expanding_window = [True, False]
    ew = False
    algos = ["xg","linreg"]
    alg = "xg"
    numest = [2,3,5,10]
    nest = 10
    maxdepth = [1,2,3,5]
    mdepth = 5
    #prediction window
    predwindow = [3,5,10,21,63,126,256]
    pw = 21
    #lookback window
    lookbacks = [256, 512, 768, 1280, 1920, 2560, 3840]
    lw = 2560
    # ticker
    #ticker = "SPY"
    #alg = "xg"
    pw = 10
    lw = 1920
    nest = 5
    mdepth = 5
    alg = "xg"
    
    
    for pw in predwindow:
        for nest in numest:
            for mdepth in maxdepth:
                for lw in lookbacks:
                    for alg in algos:
    
                            print(pw)
                            print(nest)
                            print(mdepth)
                            print(lw)
                            print(alg)
                            print(ew)
                            
                            backtest_id = str(uuid.uuid4())
                                                    
                            # Get Weekly Return
                            #test = asset.resample('W').ffill()
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
                            
                            #######
                            # FIGURE OUT WHY THIS IS HERE
                            #######

                            df = df.dropna()

                            features = ['S&P_Comp_P', 'Dividend_D', 'Earnings_E', 'Consumer_Price_Index_CPI',
                                        'Date_Fraction', 'Long_Interest_Rate_GS10', 'Real_Price',
                                        'Real_Dividend', 'Real_Total_Return_Price', 'Real_Earnings',
                                        'Real_TR_Scaled_Earnings', 'CAPE_one', 'CAPE_two', 'Excess_CAPE_Yield',
                                        'Monthly_Total_Bond_Returns', 'Real_Total_Bond_Returns',
                                        '10_Year_Annualized_Stock_Real_Return',
                                        '10_Year_Annualized_Bond_Real_Return',
                                        'Real_10_Year_Excess_Annualized_Returns', '^IRX', '^TNX', '^TYX', 'TMS',
                                        'TMS2','sma_rat', 'stvol', 'ltvol', 'vol_rat', 'p2h']
                            
                            ### Technical Features Only ###
                            
                            predf = pd.DataFrame(columns = ["pred"])
                            
                            fi_list = []
                            
                            # Create rolling/expanding window trainset
                            for i in range((lw+pw), len(df.index)-1):
                                print(i)
                                print(lw)
                                
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
                                
                                fi_list.append(pd.DataFrame({
                                    'Feature': features,
                                    'Importance': model.feature_importances_
                                }))
                                    
                                lframe = pd.DataFrame(y_pred, columns = ["pred"], index = ytest.index)
                                predf = predf.append(lframe)
                                
                                if ew == True: 
                                    lw = lw + 1
                                    
                            # Put predictions back on original data frame
                            # And convert y_pred so it can be added to dataframe
                            # Put predictions back on original data frame
                            # And convert y_pred so it can be added to dataframe
                            
                            combined_df = pd.concat(fi_list, ignore_index=True)
                            combined_df = pd.concat(fi_list, axis=1)
                            combined_df2 = pd.concat([fi_list[0]] + [df.iloc[:, 1] for df in fi_list[1:]], axis=1)
                            
                            test = combined_df2.T
                            
                            # Set the first row as the header
                            test.columns = test.iloc[0]

                            # Drop the first row as it's now the header
                            test = test.drop(test.index[0])
                            
                            # Optionally, reset the index if you want to have a continuous integer index
                            test = test.reset_index(drop=True)

                            # Plot using the plot method
                            ax = test.plot(kind='line', title='Yearly Sales', grid=True, legend=True)
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                            plt.tight_layout()

                            # Display the plot
                            plt.show()  
                            
                            # Compute the mean for all columns
                            column_means = test.mean()

                            # Sort the means in descending order
                            sorted_means = column_means.sort_values(ascending=False)
                            
                            
                            
                            sframe = df
                            predf = predf[~predf.index.duplicated()]
                            sframe['signal'] = predf
                            sframe['signal'] = sframe['signal'].shift(1)
    
                            sframe['return'] = sframe['dayret']
                            sframe = sframe.dropna()
                            sframe = sframe[~sframe.index.duplicated(keep='first')]
                            
                            # Create the strategy return performance
                            for i in range(len(sframe.index)):
                                print(i)
                                if sframe.loc[sframe.index[i], "signal"] > 0:
                                    sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*1
                                else:
                                    sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*-1
                                    
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
                            # Add accuracy: correct observations divided by all observations
                            
                            perf = pd.DataFrame({
                                'Date Run': datetime.today().strftime('%Y-%m-%d'),
                                'Backtest_id':backtest_id,
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
                            if path.exists('/home/bstaverosky/Documents/projects/MLPA/summary' + "/" + "MLPA_shiller_backtest_summary.csv") == True:
                                perf.to_csv('/home/bstaverosky/Documents/projects/MLPA/summary' + "/" + "MLPA_shiller_backtest_summary.csv", mode = 'a', header = False)
                            elif path.exists('/home/bstaverosky/Documents/projects/MLPA/summary' + "/" + "MLPA_shiller_backtest_summary.csv") == False:
                                perf.to_csv('/home/bstaverosky/Documents/projects/MLPA/summary' + "/" + "MLPA_shiller_backtest_summary.csv", header = True)
                             
                            # Save last relevant model 
                            with open("/home/bstaverosky/Documents/projects/MLPA/models/" + backtest_id + datetime.today().strftime('%Y-%m-%d') + "model.pkl", 'wb') as file:
                                pickle.dump(model, file)
                                
                            # Save Backtest Time Series
                            # if path.exists('/home/bstaverosky/Documents/projects/MLPA/backtests/backtest_ts_' + backtest_id + ".csv") == True:
                            #     strat_series.to_csv('/home/bstaverosky/Documents/projects/MLPA/backtests/backtest_ts_' + backtest_id + ".csv", mode = 'a', header = False)
                            # elif path.exists('/home/bstaverosky/Documents/projects/MLPA/backtests/backtest_ts_' + backtest_id + ".csv") == False:
                            #     strat_series.to_csv('/home/bstaverosky/Documents/projects/MLPA/backtests/backtest_ts_' + backtest_id + ".csv", header = True)
    
                            ### ADHOC FEATURE IMPORTANCE CHECK ###
                            # Assuming you've trained a model called 'model' and have a list of feature names 'feature_names'
                            feature_importance = model.feature_importances_
                            
                            feature_importances_df = pd.DataFrame({
                                'Feature': features,
                                'Importance': model.feature_importances_
                            })
                            
                            
                            
                            sorted_idx = np.argsort(feature_importance)
                            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
                            plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
                            plt.xlabel('Importance')
                            plt.title('Feature importance')
                            plt.show()
                            
                            
    
    
        
        
    
    
