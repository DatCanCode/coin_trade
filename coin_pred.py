'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function

from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv2D,Conv1D, MaxPooling2D,MaxPooling1D,Dropout, Flatten,GlobalAveragePooling1D
from keras.datasets import imdb
from keras.models import model_from_json
from keras import backend as K
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
from resnet import resnet

from utils import mfcc

import time
import dateparser
import json
import argparse
import ast
from datetime import datetime
from binance.client import Client
import rt_trade
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import math
import pickle

from scipy.stats import *#pearsonr spearmanr
from sklearn.linear_model import LinearRegression
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# DATA_INTERVAL_PER_STEP 5
# lstm_output_size = 6*3*64
COIN_FEATURE = 5
NUMCOINS=9
EXTRA_COINS=13
EXTRA_COIN_FEATURES=EXTRA_COINS*COIN_FEATURE
DF_FEATURES=COIN_FEATURE*NUMCOINS+EXTRA_COIN_FEATURES
#number of freq
 # Training
batch_size = 128
OUTPUT_SIZE=NUMCOINS*2
# epochs = 10
TIME_BETWEEN_SAMPLE = 30
seq_len=int((3*24*60)/TIME_BETWEEN_SAMPLE)
LAG_COEFF = 1
LAG_HRS = 6
LAG_TIME=int(LAG_HRS*60/TIME_BETWEEN_SAMPLE)  #5min*24 = 120min = 2hrs
# input_shape = (LAG_COEFF,12,1)
LOOK_BACK=int(6*24*60.0/TIME_BETWEEN_SAMPLE)  #5min*24 = 120min = 2hrs
LOOK_BACK_IN_MS = LOOK_BACK*TIME_BETWEEN_SAMPLE*60*1000
MFCC_RETURN_SIZE_1 = 4
PRICE_INCREASE = 1
PRICE_DECREASE = 2
PRICE_SAME = 3
PERCENT_TRIGGER = 0.015
data_dir_name  = "./data/kline/"
model_dir = "./model/"
NUM_CLASS = 4
rt = []
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk)) 
#create same size csv
def csv_preprocessing(coin_list,path):
    size=[]
    for coin in coin_list:
        df = pd.read_csv(path  + "{}.csv".format(coin))
        size.append(df.shape[0])
    size_min = min(size)
    print("size_min=",size_min)
    for coin in coin_list:
        df = pd.read_csv(path  + "{}.csv".format(coin))
        df=df.drop(df.index[0:df.shape[0]-size_min])
        # df.drop(df.index[:2], inplace=True)
        # df = df[df.shape[0]-size_min:,:]
        df.to_csv(path  + "{}.csv".format(coin))    
# join all same size csv into one        
def pd_join_csv(coin_list,path):
    cols_name =[]
    
    for coin in coin_list:
        cols_name.append("{}_High".format(coin))
        cols_name.append("{}_Low".format(coin))
        cols_name.append("{}_Close".format(coin))
        cols_name.append("{}_Volume".format(coin))
        cols_name.append("{}_Count".format(coin))

    allcoin_f= pd.DataFrame(index=None,icolumns=cols_name)
    for coin in coin_list:
        df = pd.read_csv(path  + "{}.csv".format(coin))
        allcoin_f["{}_High".format(coin)] = df['High']
        allcoin_f["{}_Low".format(coin)] = df['Low']
        allcoin_f["{}_Close".format(coin)] = df['Close']
        allcoin_f["{}_Volume".format(coin)] = df['Volume']
        allcoin_f["{}_Count".format(coin)] = df['Number of trades']
            
    allcoin_f.to_csv(path  + "allcoin.csv")    
#join all with different size csv into one        
def pd_join_diff_size_csv(coin_list,path):
    
    allcoin_f= pd.DataFrame( index=None,columns=None)
    for coin in coin_list:
        df = pd.read_csv(path  + "{}.csv".format(coin))
        allcoin_f=pd.concat([allcoin_f,df[['High','Low','Close','Volume','Number of trades']]], ignore_index=True, axis=1)
       
    print("size of join data frame ={}".format(allcoin_f.shape))
    allcoin_f.fillna(0, inplace=True)
    allcoin_f.to_csv(path  + "allcoin.csv")  

def _generate_arrays_from_file_for_training(path,look_back,lag_time,coin_list):
    while 1:
        # df = pd.read_csv(path)
        df = pd.read_csv(path  + "allcoin.csv")
        print(df.shape)
        # coin_features = df.shift(self.lag_time)
        # coin_features = coin_features.values[:-(1+self.lag_time),:]
        df_val = df.values
        pad = np.pad(df_val, (look_back,), 'edge')
        features_index=[i for i in range(df.shape[1])]
        pad_x = pad[:,features_index]

        #create labels high low close
        labels_index= [i+look_back for i in range(df.shape[1]) if (i%5==0) or (i%5==1) or 
                       (i%5==2) ]#[0+look_back,1+look_back,2+look_back]  #
        print(labels_index)
        # pad_y = np.pad(df_val, (look_back,), 'edge')
        pad_y = pad[:,labels_index]
        
        print("padx pady shape ={}{}".format(pad_x.shape,pad_y.shape))
        print(pad_y)
        for row in range(df.shape[0]):
            y = np.empty((1,OUTPUT_SIZE), float)
            tmp= np.empty((1,OUTPUT_SIZE),float)
            # create numpy arrays of input data
            # and labels, from each line in the file
            x = pad_x[row:row+look_back,:]
            
            for col in range (int(OUTPUT_SIZE/2)):
            #     y[0,col] = np.max(pad_y[row+look_back:row+look_back+lag_time,col])
            #     y[0,col+1] = np.max(pad_y[row+look_back:row+look_back+lag_time,col+1])
            #predict btc only
                min = np.mean(pad_y[row+look_back+1:row+look_back+lag_time,1+col])
                max = np.mean(pad_y[row+look_back+1:row+look_back+lag_time,0+col])
                close = np.mean(pad_y[row+look_back+1:row+look_back+lag_time,2+col])
                #increase rate
                y[0,2*col] = (max-close)/close
                #decrease rate
                y[0,2*col+1] = (close -min)/close
            # print(y)    
                # y = np.concatenate((y, tmp), axis=0)
            # y=y.reshape(-1,pad_y.shape[1])
            #reshape for tensorflow backend
            x = x.reshape(1,x.shape[1],1,x.shape[0])

            yield (x, y)
        print("generate_arrays_from_file_for_training finish")    

def generate_arrays_from_file_for_training(path,look_back,lag_time,coin_list):
    # while 1:
        from random import randint

        # df = pd.read_csv(path)
        df = pd.read_csv(path  + "allcoin.csv")
        print(df.shape)
        # coin_features = df.shift(self.lag_time)
        # coin_features = coin_features.values[:-(1+self.lag_time),:]
        df_val = df.values
        # pad = np.pad(df_val, (look_back,), 'edge')
        features_index=[i for i in range(df.shape[1])]
        pad_x = df_val[:,features_index]

        #create labels high low close
        # labels_index= [i+look_back for i in range(df.shape[1]) if (i%5==0) or (i%5==1) or 
        #                (i%5==2) ]#[0+look_back,1+look_back,2+look_back]  #
        # labels_index =  [coin_index*COIN_FEATURE ,
        #                 coin_index*COIN_FEATURE  + 1,
        #                 coin_index*COIN_FEATURE +  2]
        labels_index= [i for i in range(df.shape[1]) if (i%5==0) or (i%5==1) or 
                       (i%5==2) ]#[0+look_back,1+look_back,2+look_back]  #                
        print(labels_index)
        # pad_y = np.pad(df_val, (look_back,), 'edge')
        pad_y = df_val[:,labels_index]
        
        print("padx pady shape ={}{}".format(pad_x.shape,pad_y.shape))
        print(pad_y)
        for i in range(df.shape[0]-look_back-lag_time):
            row = randint(0, df.shape[0]-look_back-lag_time)

            y = np.empty((1,OUTPUT_SIZE), float)
            tmp= np.empty((1,OUTPUT_SIZE),float)
            # create numpy arrays of input data
            # and labels, from each line in the file
            x = pad_x[row:row+look_back,:]
            
            for col in range (int(OUTPUT_SIZE/2)):
            #     y[0,col] = np.max(pad_y[row+look_back:row+look_back+lag_time,col])
            #     y[0,col+1] = np.max(pad_y[row+look_back:row+look_back+lag_time,col+1])
            #predict btc only
                min = np.mean(pad_y[row+look_back+1:row+look_back+lag_time,1+col])
                max = np.mean(pad_y[row+look_back+1:row+look_back+lag_time,0+col])
                # close =  pad_y[row,2+col]#np.mean(pad_y[row+look_back+1:row+look_back+lag_time,2+col])
                close = np.mean(pad_y[row+look_back:row+look_back+int(lag_time/12),2+col])

                # min = pad_y[row+look_back+1:row+look_back+lag_time,1+col]
                # max = pad_y[row+look_back+1:row+look_back+lag_time,0+col]
                # close = pad_y[row+look_back+1:row+look_back+lag_time,2+col]
                #increase rate
                y[0,2*col] =  (max-close)/close
                #decrease rate
                y[0,2*col+1] =  (close -min)/close
            # print(y)    
                # y = np.concatenate((y, tmp), axis=0)
            # y=y.reshape(-1,pad_y.shape[1])
            #reshape for tensorflow backend
            x = x.reshape(1,x.shape[1],1,x.shape[0])

            yield (x, y)
        print("generate_arrays_from_file_for_training finish")    




class deep_trade():
    def __init__(self,coin_list,look_back,lag_time):
        self.coin_list = coin_list
        self.look_back = look_back
        self.lag_time = lag_time
        self.input_shape=0
        self.numoutput=0
    def encode(self,data):
        print('Shape of data (BEFORE encode): %s' % str(data.shape))
        encoded = to_categorical(data)
        print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
        return encoded
    def decode(self,datum):
        return np.argmax(datum)
   
    '''
    normalize input data with sigmoid
    '''

    def data_norm(self,x_array):
        return preprocessing.scale(x_array)
        # return x_array# preprocessing.scale(x_array)
   

    def create_binary_labels(self,labels, lag_time):
        a = np.zeros((labels.shape[0])) #,lag_time,labels.shape[1]))
        #padd lag_time rows to labels
        pad_a = np.pad(labels, (lag_time,), 'edge')
        print(labels.shape)
        print(pad_a.shape)
        for n in range(labels.shape[0]):
            rate = (np.max(pad_a[n+1:n+lag_time,0])-pad_a[n,1])/pad_a[n,1]
            if  rate > PERCENT_TRIGGER:
                a[n] = PRICE_INCREASE
            elif rate < -PERCENT_TRIGGER: 
                a[n] = PRICE_DECREASE
            else :
                a[n] = PRICE_SAME   
        return encode(a)
    
        
    def create_input(self,x_in ,look_back):
        # a = np.zeros((x_in.shape[0],2)) #,look_back,labels.shape[1]))
        pad_a = np.pad(x_in, (look_back,), 'edge')
        f=[]
        print("x_in shape =",x_in.shape)
        for row in range(x_in.shape[0]):

            a = pad_a[row:row+look_back,:]
            f.append(a)
            # print ('ret shape:',len(ret)) 

        ret = np.array(f).reshape(x_in.shape[0],look_back,-1)  
        print ('ret shape:',ret.shape) 
        print ('x_in shape:',x_in.shape) 
        return ret

    def build_model(self,input_shape ,numoutput):

        DIM_ORDERING = {'th', 'tf'}

        model = resnet.ResnetBuilder.build_resnet_18(input_shape, numoutput)
        for ordering in DIM_ORDERING:
            K.set_image_dim_ordering(ordering)
            adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            # model.compile(loss="mse", optimizer=adam)
            model.compile(loss="mse", optimizer=adam, metrics=['mse'])

            assert True, "Failed to compile with '{}' dim ordering".format(ordering)
        return model

    def neuralnet_train(self,mode,epochs):
        

        print('Build model...')
        model={}
        if(mode == 'TRAIN_CONTINUE'):
            json_file = open("model/model_all_coin.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("model/model_all_coin.h5")
            print("Loaded model from disk")
        else :
            # model = build_model([1,DF_FEATURES* LAG_COEFF,seq_len, 2])
            model = self.build_model(input_shape = [LOOK_BACK,DF_FEATURES,1],numoutput=OUTPUT_SIZE)#*len(self.coin_list))
            # evaluate loaded model on test data
        adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='mse', optimizer=adam, metrics=['mse'])
        
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print("> Compilation Time : ", time.time() - start)
    
        # print('Train for {}...'.format(coin))
        # model.fit(x_train, y_train,
        #         batch_size=batch_size,
        #         epochs=epochs,
        #         validation_data=(x_test, y_test))
        
        model.fit_generator(generate_arrays_from_file_for_training(data_dir_name,self.look_back,self.lag_time,self.coin_list),
                samples_per_epoch=500, nb_epoch=epochs)
        
        
        # score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        # print('Test score:', score)
        # print('Test accuracy:', acc)
        # with open("accuracies.txt", "a") as acc:
        #     acc.write("coin {} || test score {} || test accuracy : {}".format(coin, score,acc))
        
        # serialize model to JSON
        model_json = model.to_json()
        with open("model/model_all_coin.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model/model_all_coin.h5")
        print("Saved model to disk")
        
       
    def get_kline_lag_time(self,coin , lookback_in_ms):

        # lookback = 2*60*1000 #5mins
        delay = 5*60*1000 #5mins
        rt.coin_list=[coin]

        end_ts = rt.get_time()
        #  calendar.timegm(time.gmtime()) -lookback

        start_ts = end_ts - lookback_in_ms
        print("start=",start_ts)
        print("end=",end_ts)

        f = rt.get_historical_klines(symbol=coin, 
                        interval =Client.KLINE_INTERVAL_30MINUTE,
                        end=end_ts,
                        start=start_ts)
        f = ast.literal_eval(json.dumps(f))
        return f
    def create_predictive_input(self):

        features = np.empty((self.look_back,0), float)
        coin_f_index=[2,3,4,5,7]
        for coin in self.coin_list:
            coin_f = np.array(self.get_kline_lag_time(coin,LOOK_BACK_IN_MS),dtype =np.float).reshape(-1,12)
            coin_f = coin_f[:,coin_f_index]
            # print("coin_f shape ",coin_f.shape)

            # COIN_FEATURE = np.concatenate((COIN_FEATURE, tmp), axis=0)
            features = np.hstack((features,coin_f))
        # features = self.create_input(features,self.look_back)
            # DF_FEATURES
         #reshape for tensorflow backend
        features = features.reshape(1,DF_FEATURES,1,self.look_back)

        print("create_predictive_input, features shape ",features.shape)
        
        '''
        [[[1539347100000, '0.01168000', '0.01168000', '0.01168000', '0.01168000', '40353.90000000', 1539347399999, '471.33355200', 3, '0.00000000', '0.00000000', '0']], 
        [[1539347100000, '0.50330000', '0.50410000', '0.50220000', '0.50280000', '24566.00000000', 1539347399999, '12359.97658800', 48, '10247.93000000', '5157.88996800', '0']]]

        '''
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
        # print("kline shape after modify",features.shape)
        # features = self.create_input(features ,LOOK_BACK)
        return features


    def neuralnet_out(self,input_x):

        # load json and create model
        json_file = open('model/model_all_coin.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model/model_all_coin.h5")
        print("Loaded model from disk")
        
        # evaluate loaded model on test data
        # loaded_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        loaded_model.compile(loss='mse', optimizer=adam, metrics=['mse'])

        # score = loaded_model.evaluate(X, Y, verbose=0)
    # score = loaded_model.evaluate(X, Y, verbose=0)
        
        prediction = loaded_model.predict(input_x)
        # out =prediction

        print(prediction)
        x = input_x.reshape(-1,DF_FEATURES)
        prGreen ("COIN : {}".format(coin))

        prediction = prediction.flatten()
        
        for i ,coin in enumerate(self.coin_list):
            close = x[-2,i*5+2]
            print("close price =",close)
            # if(prediction[2*i] > 0):
            h =(prediction[2*i+0] + 1)*close
            l =(-prediction[2*i+1] + 1)*close
            t = LAG_TIME*TIME_BETWEEN_SAMPLE/60
            #print("COIN : {} ".format(coin,l,h,t))
            prGreen("mean HIGH".format(h))
            prGreen("mean LOW".format(l))
            
        return prediction
    
    def logic_trade(self):
        '''
            get all the coin predict values, the determine buy or shell based on current account info and predicted values.
            profit=0
            if price increase :
                if ishold(coin)
                    keep holding
                else:
                    buy(coin)    
            elif price decrease:
                if ishold(coin)
                    shell(coin)
                else:
                    do nothing   
            else :
                do nothing    

            if price drop > 10%

        '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decide to Train or to Run.')
    parser.add_argument('--mode', type=str, help='--mode=train or --mode=trainor --mode=predict')
    parser.add_argument('--coin', type=str, help='--coin=BTCUSDT')
    parser.add_argument('--key', type=str, help='--coin=BTCUSDT')
    parser.add_argument('--pw', type=str, help='--coin=BTCUSDT')
    
    args = parser.parse_args()
    extra_list =  '''
    ETHBTC
    XRPBTC
    BCCBTC
    BNBBTC
    LTCBTC   
    NEOBTC    
    ZRXBTC
    ADABTC
    QKCBTC
    IOTABTC
    MDABTC
    BATBTC
    EOSBTC
    '''.split() 
    predict_list =  '''
    BTCUSDT
    ETHUSDT
    XRPUSDT
    BCCUSDT
    BNBUSDT
    LTCUSDT   
    NEOUSDT    
    QTUMUSDT
    ADAUSDT
    '''.split()
    total_list = predict_list+extra_list
    # symbol_list =  "IOTAUSDT".split()
    if (args.coin):
        predict_list =  args.coin.split()
    print ("mode=",args.mode)
    rt=rt_trade.trade_realtime(api_key=args.key, api_secret=args.pw)

    # for symbol in symbol_list:
    #     print("coin  =",symbol)
    trade = deep_trade(predict_list,LOOK_BACK,LAG_TIME)
    #preprocessing data if needed
    # csv_preprocessing(symbol_list,data_dir_name)
    if args.mode == "train":   

        trade.neuralnet_train('train',10)
    elif args.mode == "conti":  
        trade.neuralnet_train('TRAIN_CONTINUE',5)
    elif args.mode == "predict":  

        input_x = trade.create_predictive_input()   #create_lstm_input(symbol)
        # rt.coin_list = 
        for index,coin in enumerate(trade.coin_list):
            trade.load_model(index ,coin)
        print("input shape =",input_x.shape)
        total = trade.neuralnet_out(input_x)
    elif args.mode=="processdata":
        csv_preprocessing(total_list,data_dir_name)
        pd_join_csv(total_list,data_dir_name)
      #  pd_join_diff_size_csv(symbol_list,data_dir_name)

        # print(decode(total))
        # print(rt.get_asset_balance(asset='USDT'))
        # print(rt.get_account())            
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Decide to Train or to Run.')
#     parser.add_argument('--mode', type=str, help='--mode=train or --mode=trainor --mode=predict')
#     parser.add_argument('--coin', type=str, help='--coin=BTCUSDT')
#     parser.add_argument('--key', type=str, help='--coin=BTCUSDT')
#     parser.add_argument('--pw', type=str, help='--coin=BTCUSDT')
    
#     args = parser.parse_args()
#     symbol_list =  '''
#     BTCUSDT
#     ETHUSDT
#     ETHBTC
#     XRPBTC
#     XLMBTC
#     EOSBTC
#     BCCUSDT
#     BNBUSDT
#     LTCUSDT   
#     NEOUSDT    
#     TRXBTC
#     ZRXBTC
#     ARNBTC
#     '''.split()
#     # symbol_list =  "IOTAUSDT".split()
#     if (args.coin):
#         symbol_list =  args.coin.split()
#     print ("mode=",args.mode)
#     rt=rt_trade.trade_realtime(api_key=args.key, api_secret=args.pw)

#     # for symbol in symbol_list:
#     #     print("coin  =",symbol)
#     trade = deep_trade(symbol_list,LOOK_BACK,LAG_TIME)
#     #preprocessing data if needed
#     # csv_preprocessing(symbol_list,data_dir_name)
#     if args.mode == "train":   

#         trade.neuralnet_train('train',3)
#     elif args.mode == "conti":  
#         trade.neuralnet_train('TRAIN_CONTINUE',3)
#     elif args.mode == "predict":  

#         input_x = trade.create_predictive_input()   #create_lstm_input(symbol)
#         # rt.coin_list = 

#         print("input shape =",input_x.shape)
#         total = trade.neuralnet_out(input_x)
#     elif args.mode=="processdata":
#         csv_preprocessing(symbol_list,data_dir_name)
#         pd_join_csv(symbol_list,data_dir_name)
#       #  pd_join_diff_size_csv(symbol_list,data_dir_name)

#         # print(decode(total))
#         # print(rt.get_asset_balance(asset='USDT'))
#         # print(rt.get_account())            