import time
import dateparser
import json

from datetime import datetime
from binance.client import Client
import rt_trade
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import math

from scipy.stats import *#pearsonr spearmanr
from sklearn.linear_model import LinearRegression
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import argparse

import pickle
# EOSUSDT = 'EOSUSDT'

dir_name = "./data/kline/"
model_dir = "./model/"
LAG_COEFF = 240 #5min*24 = 120min = 2hrs


def get_ticker():
    client = Client("", "")
    data = client.get_ticker()
    with open(
            "btc_ticker.py",
            'w'  # set file write mode
        ) as f:
            f.write("btc_ticker=" + json.dumps(data))

def train(coin):
    df0 = pd.read_csv(dir_name + "{}.csv".format(coin))
    '''
        [
            [
                1499040000000,      # Open time
                "0.01634790",       # Open
                "0.80000000",       # High
                "0.01575800",       # Low
                "0.01577100",       # Close
                "148976.11427815",  # Volume
                1499644799999,      # Close time
                "2434.19055334",    # Quote asset volume
                308,                # Number of trades
                "1756.87402397",    # Taker buy base asset volume
                "28.46694368",      # Taker buy quote asset volume
                "17928899.62484339" # Can be ignored
            ]
        ]
    '''
    #[high low volume  quoteasseet count  ]
    

    #np.delete(coin_features, [0,1,2,3,4,6,9,10,11], 1)

    df = df0.drop(['Open time', 'Open','Volume','Number of trades','Close time','Quote asset volume',
                            'Taker buy base asset volume','Taker buy quote asset volume','ignore'], axis=1)

    y = df0['Close'].values
    coin_price=y
    # coin_price = np.average(y,axis=1) #= np.subtract(y1,y2)#,axis=1)
    coin_features = df
    # df['y'] = df['Open']

    df = df.shift(LAG_COEFF)
    # df = df.Low.shift(4)

    # df = df.dropna(axis=0, how='all')
    # pd.DataFrame(df).fillna(0)
    df.fillna(0, inplace=True)
    coin_features = df.values#as_matrix(columns=df.columns[:2])

    # coin_price = df.as_matrix(columns=df.columns[2])
    print("shape coin price: ",coin_price.shape)

    print("size of {} coin_features:".format(coin),len(coin_features))


    correlations=[]
    for i in range(coin_features.shape[1]):
        correlations.append(spearmanr(coin_features[:,i],coin_price)[0])

    print(correlations)

    # Train a simple linear regression model
    regr = linear_model.LinearRegression()

    coin_features=np.array(coin_features)
    print(coin_features.shape)
    X= coin_features #new_data.values
    y = coin_price #data.price.values

    # X=np.swapaxes(X,0,1)
    #y=np.swapaxes(b,0,1)

    print("X=",X.shape)
    print("y=",y.shape)


    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)
    regr.fit(X_train, y_train)
    # print(regr.predict(X_test))
    regr.score(X_test,y_test)

    # Calculate the Root Mean Squared Error
    print("RMSE: %.2f"
        % math.sqrt(np.mean(np.abs(regr.predict(X_test) - y_test))))
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                            colsample_bytree=1, max_depth=7)
    traindf, testdf = train_test_split(X_train, test_size = 0.3)
    xgb.fit(X_train,y_train)
    pickle.dump(xgb, open("{}xgb.{}.dat".format(model_dir,coin), "wb"))
    # print(X_test)
    predictions = xgb.predict(X_test)
    print("test accuracy = ",explained_variance_score(predictions,y_test))

def predict(coin):

    xgb = pickle.load(open("{}xgb.{}.dat".format(model_dir,coin), "rb"))

    rt = rt_trade.trade_realtime(api_key='', api_secret='',coin_list=[coin])

    lookback = 2*60*1000 #5mins
    delay = 5*60*1000 #5mins

    end_ts = rt.get_time() -lookback
    #  calendar.timegm(time.gmtime()) -lookback

    start_ts = end_ts - delay
    print("start=",start_ts)
    print("end=",end_ts)

    # print(kl.get_trade(start=start_ts,end=end_ts))

    f =  rt.get_kline(start=start_ts,end=end_ts)
   
    remove_index = [0,4,5,6,7,8,9,10,11]
    '''
        [
            [
         b       1499040000000,      # Open time
         1       "0.01634790",       # Open
          2      "0.80000000",       # High
          3      "0.01575800",       # Low
          4      "0.01577100",       # Close
          5      "148976.11427815",  # Volume
         b       1499644799999,      # Close time
          6      "2434.19055334",    # Quote asset volume
          7      308,                # Number of trades
          b      "1756.87402397",    # Taker buy base asset volume
          b      "28.46694368",      # Taker buy quote asset volume
          b      "17928899.62484339" # Can be ignored
            ]
        ]
    '''

    features = np.array(f,dtype=np.float).flatten()
    features = np.delete(features, remove_index).reshape((1,3))

    #[high low volume quoteasseet  count ]
    print(features.shape)

                #  [6568, 11.16, 227.12]])
    # features=np.swapaxes(features,0,1)
    print ("test voi btc features hien tai :")
    print(features)
    print(features.shape)
    predictions = xgb.predict(features)
    print(predictions.shape)
    print ("gia {} coin du kien trong vong 1 tieng = {} ".format(coin,predictions))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decide to Train or to Run.')
    parser.add_argument('--mode', type=str, help='--mode=train or --mode=trainor --mode=predict')
    parser.add_argument('--coin', type=str, help='--coin=BTCUSDT')

    args = parser.parse_args()
    symbol_list_ =  '''
    BNBUSDT
    EOSUSDT
    BCCUSDT
    LTCUSDT
    IOTAUSDT
    NEOUSDT
    ADAUSDT
    YOYOBTC
    VETUSDT
    ETCUSDT
    ONTUSDT
    XRPUSDT 
    ETHUSDT
    TRXUSDT
    '''.split()
    if (args.coin != ''):
        symbol_list =  args.coin.split()
    print ("mode=",args.mode)
    for symbol in symbol_list:
        print("coin  =",symbol)

        if args.mode == "train":
            train(symbol)
        elif args.mode == "predict":  
            predict(symbol)