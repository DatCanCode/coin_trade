
import threading
import time
import os.path

import time
import dateparser
import json
import calendar
from datetime import datetime
from binance.client import Client
from agent.agent import Agent
# from functions import *
import sys
import pandas as pd
import numpy as np
import time
import argparse
from keras.models import load_model
import ast
from datetime import datetime
dir_name = "./data/kline/"
import pickle 
NORMAL_TRADE_ID = 0
DANGEROUS_SIG_ID = 1
num_trade = 10000
# DATA_INTERVAL_PER_STEP 5
# lstm_output_size = 6*3*64
COIN_FEATURE = 5
NUMCOINS=22
NUM_PREDICT_COIN=1
DF_FEATURES=COIN_FEATURE*NUMCOINS

# Training
# epochs = 10
TIME_BETWEEN_SAMPLE = 30
seq_len=int((3*24*60)/TIME_BETWEEN_SAMPLE)
LAG_COEFF = 1
# input_shape = (LAG_COEFF,12,1)
LOOK_BACK=int(10*24*60.0/TIME_BETWEEN_SAMPLE)  #5min*24 = 120min = 2hrs
LOOK_BACK_IN_MS = (LOOK_BACK+8)*TIME_BETWEEN_SAMPLE*60*1000
batch_size = 64
train_steps = 16
data_dir_name  = "./data/kline/"
model_dir = "./model/"
trade={}
agent={}
l=5500
def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds

    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str

    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms
def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds

    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/

    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)

class trade_realtime():

    def __init__(self, api_key, api_secret, total_list=None):
        self.coin_list = total_list
        self.api_key = api_key
        self.api_secret = api_secret
        self.client= Client(api_key, api_secret)
    def get_time(self):
        server_time = self.client.get_server_time()
        #  {
        #         "serverTime": 1499827319559
        # }
        return int(server_time['serverTime'])
    def get_exchange_status(self):
        return self.client.get_system_status()
   
    def get_coin_price(self,coin_list):
        # kwargs = {'data': coin}
        output_data = []
        for coin in coin_list:
            price_d =  ast.literal_eval(json.dumps(self.client.get_symbol_ticker(symbol =coin))) 
            print(price_d)  
            price = float(price_d['price'])
            output_data.append(price)

        return output_data
    def get_trade(self , start,end):
        output_data = []
        for coin in self.coin_list:
            output_data.append(self.client.get_aggregate_trades(symbol = coin,
                    startTime=start,endTime=end))    

        return output_data
    def get_kline(self ,start,end):

        output_data = []
        for coin in self.coin_list:
            output_data.append(self.client.get_klines(
                symbol= coin,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                limit=500,
                startTime=start,
                endTime=end
            ))
        return output_data
        
    def get_historical_klines(self,symbol, interval,start, end):
       
        # init our list
        output_data = []

        # setup the max limit
        limit = 500
        timeframe = interval_to_milliseconds(interval)
        start_ts=start
        idx = 0
        # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
        symbol_existed = False
        while True:
            # fetch the klines from start_ts up to max 500 entries or the end_ts if set
            temp_data = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=start_ts,
                endTime=end
            )

            # handle the case where our start date is before the symbol pair listed on Binance
            if not symbol_existed and len(temp_data):
                symbol_existed = True

            if symbol_existed:
                # append this loops data to our output data
                output_data += temp_data

                # update our start timestamp using the last value in the array and add the interval timeframe
                start_ts = temp_data[len(temp_data) - 1][0] + timeframe
            else:
                # it wasn't listed yet, increment our start date
                start_ts += timeframe

            idx += 1
            # check if we received less than the required limit and exit the loop
            if len(temp_data) < limit:
                # exit the while loop
                break

            # sleep after every 3rd call to be kind to the API
            if idx % 3 == 0:
                time.sleep(1)

        return output_data
    def get_orderbook_ticker(self):
        pass
    def order_limit_buy(self,  **params):
      
        return self.client.order_limit_buy(**params)
    def order_limit_sell(self, **params):
       
        return self.client.order_limit_sell( **params)
    def order_market_sell(self,**params):
        return self.client.order_market_sell( **params)  
    def order_market_buy(self,**params):
        return self.client.order_market_buy( **params)  
    def get_open_orders(self,**params):
        return self.client.get_open_orders(**params)
     
    def create_test_order(self, **params):

        self.client.create_test_order()

    def get_order(self, **params):
        self.client.get_order(self, **params)
    def get_all_orders(self, **params):
        self.client.get_all_orders(self, **params)
    def cancel_order(self, **params):
        self.client.cancel_order(self, **params)
    def get_account(self, **params):
        return(self.client.get_account(recvWindow=self.get_time()))
    def get_asset_balance(self, asset, **params):
        bal = self.client.get_asset_balance(asset=asset,recvWindow=self.get_time())
        return ast.literal_eval(json.dumps(bal))['free']
    def start_trade():
        pass
    def get_kline_lag_time(self, coin , lookback_in_ms):

            # lookback = 2*60*1000 #5mins
            # rt.pred_coin_list=[coin]

            end_ts = self.get_time()
            #  calendar.timegm(time.gmtime()) -lookback

            start_ts = end_ts - lookback_in_ms
          #  print("start=",start_ts)
           # print("end=",end_ts)

            f = self.get_historical_klines(symbol=coin, 
                            interval =Client.KLINE_INTERVAL_30MINUTE,
                            end=end_ts,
                            start=start_ts)
            f = ast.literal_eval(json.dumps(f))
            return f
    def getState(self,coin_list):

        features = np.empty((LOOK_BACK,0), float)
        coin_f_index=[2,3,4,5,7]
        for coin in coin_list:
            coin_f = np.array(self.get_kline_lag_time(coin,LOOK_BACK_IN_MS),dtype =np.float).reshape(-1,12)
            coin_f = coin_f[coin_f.shape[0]-LOOK_BACK:,coin_f_index]
            if (coin_f.shape[0] <10 ):
                print("something is wrong with binance api,return shape=",coin_f.shape)
                return
            #print("coin_f shape ",coin_f.shape)
            #print("features shape ",features.shape)

            # COIN_FEATURE = np.concatenate((COIN_FEATURE, tmp), axis=0)
            features = np.hstack((features,coin_f))
        # features = self.create_input(features,LOOK_BACK)
            # DF_FEATURES
         #reshape for tensorflow backend
        features = features.reshape(1,DF_FEATURES,1,LOOK_BACK)

        print("create_predictive_input, features shape ",features.shape)
       
        # print("kline shape after modify",features.shape)
        # features = self.create_input(features ,LOOK_BACK)
        return features 
class trade_thread(threading.Thread):
    def __init__(self, threadID='', name='' , total_list='',model ="",agent = "",
        api_key='', api_secret='',coin = 'XRPUSDT'):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.coin = coin
        self.coin_name = 'XRP'
        self.total_list = total_list
        self.model = model
        self.agent = agent
        self.api_key = args.key
        self.api_secret =args.pw
        self.exchange = trade_realtime(api_key = api_key,api_secret=api_secret,  total_list=self.total_list)
        self.wait = 30
    def auto_trade(self,coin):
        # thread_check(self.name, 5, self.counter)
        coin_fund = 11 #$
        usdt_fund = 100
        total_profit=0
        sav_count = 0

        #test exchange
        print(self.exchange.get_asset_balance(asset='USDT'))
        for c in total_list:
            self.agent.inventory[c] = []
            print("agent.inventory[{}]={}".format(c,self.agent.inventory[c]))
        #get asset balance
        num_coin = float(self.exchange.get_asset_balance (asset = self.coin_name))
        print("coin {} balance = {}".format(self.coin,num_coin))
        price = self.exchange.get_coin_price(coin_list = [coin])[0]
        for i in range (int(num_coin/27)):
                self.agent.inventory[self.coin].append([27, price])
                #self.agent.inventory[self.coin].append([num_coin, price])
        #self.agent.inventory[self.coin].append([27,0.46709])
        state = self.exchange.getState(self.total_list)    
        for t in range(num_trade):
            action = self.agent.act(state)
            print("action =",action)
            # sit
            next_state = self.exchange.getState(self.total_list)
            time.sleep(30)
            print("state shape =",next_state.shape)

            price = self.exchange.get_coin_price(coin_list = [coin])[0]
           # price_d = ast.literal_eval(json.dumps(price_d))
            print(price)
            # price = 
            print("{} price {}".format(self.coin,price))

            reward = 0
            if action == 1 and usdt_fund > 0: # buy coin i-th
                # print("price=",price)
                num_coin = int(coin_fund/price)
              
                ret = self.exchange.order_limit_buy(symbol=coin , price = price, quantity = num_coin)

                #check return befor continue
                #just wait and check coin account
                # while(self.wait ):
                #     order = self.exchange.get_open_orders(symbol=coin)
                #     if(order['symbol'] == coin and order['status'] == "NEW"):
                #         print("fail to place order")
                #         time.sleep(5) #wait 5 seconds
                #         self.wait -= 1
                #     else:                          
                #         self.agent.inventory[coin].append([num_coin,price])
                #         print ("Buy: {} of {} at price {} ".format(coin,num_coin,price))
                #         usdt_fund -= coin_fund
                #         reward -= 0.001*coin_fund
                #         break
                self.agent.inventory[coin].append([num_coin,price])
                print ("Buy: {} of {} at price {} ".format(coin,num_coin,price))
                usdt_fund -= coin_fund
                reward -= 0.001*coin_fund        
            elif action == 2 and len(self.agent.inventory[coin]) > 0: # sell
                print("agent.inventory[{}]={}".format(coin,self.agent.inventory[coin]))
                
                num_coin , bought_price = self.agent.inventory[coin].pop(0)   
                ret = self.exchange.order_limit_sell(symbol=coin , price = price, quantity = num_coin)
                print("return command",ret)
                # while(self.wait ):
                #     order = self.exchange.get_open_orders(symbol=coin)
                #     if(order['symbol'] == coin and order['status'] == "NEW"):
                #         print("fail to place order")
                #         time.sleep(2) #wait 5 seconds
                #         self.wait -= 1
                #     else:                          
                #         self.agent.inventory[coin].append([num_coin,price])
                #         print ("Buy: {} of {} at price {} ".format(coin,num_coin,price))
                #         usdt_fund -= coin_fund
                #         reward -= 0.001*coin_fund
                #         break
                coin_fund  = num_coin*price*0.999     
                usdt_fund += coin_fund             
                reward += (price - bought_price)*num_coin*0.999
                total_profit += num_coin*(price - bought_price)
                print ("Sell: {} | Profit: {}".format(coin,num_coin*(price - bought_price)))
                # print ("Buy: {} of {} at price {} ".format(coin,num_coin,price))

            # reward = sum(reward)
            print("reward=",reward)
            print("usdt_fund=",usdt_fund)
            print("total_profit:",total_profit)
            done = True if t == l - 1 else False

            state = next_state
            if done:
                print ("--------------------------------")
                print ("Total Profit: {}".format(total_profit))
                print ("--------------------------------")
            
            if len(agent.memory) %(train_steps) == 0 and len(agent.memory) > batch_size:
                print("back propagation")
                agent.expReplay(batch_size)
                sav_count += 1
                if sav_count == 10:
                    sav_count = 0
                    # pickle.dump(agent,"model/model_tmp")
                    pickle.dump(agent, open("{}_agent.{}.dat".format(model_dir,coin), "wb"))

    def check_price_drop(self):
        pass
    def start(self):
        if self.threadID == NORMAL_TRADE_ID:
            self.auto_trade(self.coin)
        elif self.threadID == DANGEROUS_SIG_ID:    
            self.check_price_drop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decide to Train or to Run.')
    parser.add_argument('--mode', type=str, help='--mode=trade or --mode=run')
    parser.add_argument('--coin', type=str, help='--coin=BTCUSDT')
    parser.add_argument('--key', type=str, help='--coin=BTCUSDT')
    parser.add_argument('--pw', type=str, help='--coin=BTCUSDT')
    args = parser.parse_args()

    total_list =  '''
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

    # total_list = predict_list+extra_list
    predict_list= ['XRPUSDT']
    if (args.coin):
        symbol_list =  args.coin.split()
    model_name = 'model_tmp'
    if os.path.isfile("{}_agent.{}.dat".format(model_dir,predict_list[0])):
        agent = pickle.load(open("{}_agent.{}.dat".format(model_dir,predict_list[0]), "rb"))
    else:
        model = load_model("model/" + model_name)
        agent = Agent([LOOK_BACK,DF_FEATURES,1],NUM_PREDICT_COIN*3,is_eval=True,model_name="model_tmp")

  #  trade = coin_pred.deep_trade(predict_list,total_list,LOOK_BACK,LAG_TIME)

    auto_trade_thread = trade_thread(threadID=NORMAL_TRADE_ID,name= "auto-trade",total_list = total_list,
          model=model,agent=agent,  api_key=args.key, api_secret=args.pw,coin = 'XRPUSDT')
    auto_trade_thread.start()        
    # urgent_thread = trade_thread(DANGEROUS_SIG_ID, "normal-trading", 2)

    # for index,coin in enumerate(trade.pred_coin_list):
    #     trade.load_model(index ,coin)
    
