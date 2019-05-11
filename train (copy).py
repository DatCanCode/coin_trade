from agent.agent import Agent
# from functions import *
import sys
import coin_pred as coin_pred
import pandas as pd
import numpy as np

# DATA_INTERVAL_PER_STEP 5
# lstm_output_size = 6*3*64
COIN_FEATURE = 5
NUMCOINS=9
NUM_PREDICT_COIN=1
EXTRA_COINS=13
EXTRA_COIN_FEATURES=EXTRA_COINS*COIN_FEATURE
DF_FEATURES=COIN_FEATURE*NUMCOINS+EXTRA_COIN_FEATURES

# Training
# epochs = 10
TIME_BETWEEN_SAMPLE = 30
seq_len=int((3*24*60)/TIME_BETWEEN_SAMPLE)
LAG_COEFF = 1
LAG_TIME_HRS = 6
LAG_TIME=int(LAG_TIME_HRS*60/TIME_BETWEEN_SAMPLE)  #5min*24 = 120min = 2hrs
# input_shape = (LAG_COEFF,12,1)
LOOK_BACK=int(6*24*60.0/TIME_BETWEEN_SAMPLE)  #5min*24 = 120min = 2hrs
LOOK_BACK_IN_MS = (LOOK_BACK+8)*TIME_BETWEEN_SAMPLE*60*1000
batch_size = 32

data_dir_name  = "./data/kline/"
model_dir = "./model/"
traed={}
agent={}
l=5500

def getState(data_gen):
    
    x, y = next(data_gen)
    # state = trade.get_cnn_out(train_data).reshape(1,-1)
    # print("state shape = {}".format(state.shape))
    # print("y shape = {}".format(y.shape))
    return x, y

def generate_arrays_from_file_for_training(path,look_back,lag_time):
    #while 1:
        # df = pd.read_csv(path)
        df = pd.read_csv(path  + "allcoin.csv")
        print(df.shape)
        # coin_features = df.shift(self.lag_time)
        # coin_features = coin_features.values[:-(1+self.lag_time),:]
        df_val = df.values
        # pad = np.pad(df_val, (look_back,), 'edge')
        features_index=[i for i in range(df.shape[1])]
        pad_x = df_val[:,features_index]
	
        print("padx shape = {} ".format(pad_x.shape ))
        for row in range(0,df.shape[0]-look_back-lag_time,int(lag_time/6)):
            y = [] # np.empty((1,1), float)
            # tmp= np.empty((1,NUM_PREDICT_COIN),float)
            # create numpy arrays of input data
            # and labels, from each line in the file
            x = pad_x[row:row+look_back,:]
            # x = x.reshape(1,x.shape[1],1,x.shape[0])
            x = x.reshape(1,DF_FEATURES,1,look_back)
            
            for col in range (NUM_PREDICT_COIN):       
               
                close = np.mean(pad_x[row+look_back:row+look_back+int(lag_time/12),2+col*COIN_FEATURE])  
                # close = np.mean(pad_x[row+look_back+1:row+look_back+int(lag_time/12),2+col*COIN_FEATURE]).reshape(1,-1)
                # y = np.hstack((y,close))
                # print(close)
                y.append(close)
            yield (x, np.array(y).flatten())
        
        print("generate_arrays_from_file_for_training finish")    


def deepq_train(predict_list,episode_count):
    coin_fund = 100
    fund=1000

    for e in range(episode_count + 1):
        data_gen = generate_arrays_from_file_for_training(data_dir_name,LOOK_BACK,LAG_TIME)
        print ("Episode " + str(e) + "/" + str(episode_count))
        state , price = getState(data_gen)
        total_profit = 0
        print(predict_list)
        for coin in predict_list:
            agent.inventory[coin] = []
            print("agent.inventory[{}]={}".format(coin,agent.inventory[coin]))

        for t in range(l):
            action = agent.act(state)
            print("action =",action)
            # sit
            next_state, price = getState(data_gen)
            reward = []
            for i , coin in enumerate(predict_list):
                if action[i] == 1: # buy coin i-th
                    # print("price=",price)
                    num_coin = coin_fund/price[i]
                    agent.inventory[coin].append([num_coin,price[i]])
                    print ("Buy: {} of {} at price {} ".format(coin,num_coin,price[i]))
                    # fund -=coin_fund
                elif action[i] == 2 and len(agent.inventory[coin]) > 0: # sell
                    print("agent.inventory[{}]={}".format(coin,agent.inventory[coin]))
                    num_coin , bought_price = agent.inventory[coin].pop(0)   
                    coin_fund  = num_coin*price[i]                  
                    reward.append(num_coin*(price[i] - bought_price))
                    total_profit += num_coin*(price[i] - bought_price)
                    print ("Sell: {} | Profit: {}".format(coin,num_coin*(price[i] - bought_price)))
                    print ("Buy: {} of {} at price {} ".format(coin,num_coin,price[i]))

            reward = max(sum(reward), 0)
                    
            print("total_profit:",total_profit)
            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            # agent.memory.append((state, action, reward, next_state, done))

            state = next_state
            if done:
                print ("--------------------------------")
                print ("Total Profit: {}".format(total_profit))
                print ("--------------------------------")

            if len(agent.memory) > batch_size:
                print("back propagation")
                agent.expReplay(batch_size)

        if e % 10 == 0:
            agent.model.save("model/model_ep" + str(e))
if __name__ == "__main__":
    episode_count = int(sys.argv[1])

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
    agent = Agent([LOOK_BACK,DF_FEATURES,1],NUM_PREDICT_COIN*3)
    
  #  trade = coin_pred.deep_trade(predict_list,total_list,LOOK_BACK,LAG_TIME)

    # for index,coin in enumerate(trade.pred_coin_list):
    #     trade.load_model(index ,coin)
    deepq_train(predict_list,episode_count)

        
        