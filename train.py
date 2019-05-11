from agent.agent import Agent
# from functions import *
import sys
import pandas as pd
import numpy as np
import argparse
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
# LAG_TIME=int(LAG_TIME_HRS*60/TIME_BETWEEN_SAMPLE)  #5min*24 = 120min = 2hrs
# input_shape = (LAG_COEFF,12,1)
LOOK_BACK=int(10*24*60.0/TIME_BETWEEN_SAMPLE)  #5min*24 = 120min = 2hrs
LOOK_BACK_IN_MS = (LOOK_BACK+8)*TIME_BETWEEN_SAMPLE*60*1000
batch_size = 64
train_steps = 16
data_dir_name  = "./data/kline/"
model_dir = "./model/"
traed={}
agent={}
shift =4500
l=6560-LOOK_BACK-shift
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk)) 
def getState(data_gen):
    
    x, y = next(data_gen)
    # state = trade.get_cnn_out(train_data).reshape(1,-1)
    # print("state shape = {}".format(state.shape))
    # print("y shape = {}".format(y.shape))
    return x, y

def generate_arrays_from_file_for_training(path,look_back,coin_index):
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
        for row in range(shift,df.shape[0]-look_back):
            y = [] # np.empty((1,1), float)
            # tmp= np.empty((1,NUM_PREDICT_COIN),float)
            # create numpy arrays of input data
            # and labels, from each line in the file
            x = pad_x[row:row+look_back,:]
            # x = x.reshape(1,x.shape[1],1,x.shape[0])
            x = x.reshape(1,DF_FEATURES,1,look_back)
            
            for col in range (NUM_PREDICT_COIN):       
               
                close = np.mean(pad_x[row+look_back:row+look_back+1,2+coin_index*COIN_FEATURE])  
                # close = np.mean(pad_x[row+look_back+1:row+look_back+int(lag_time/12),2+col*COIN_FEATURE]).reshape(1,-1)
                # y = np.hstack((y,close))
                # print(close)
                y.append(close)
            yield (x, np.array(y).flatten())
        
        print("generate_arrays_from_file_for_training finish")    


def deepq_train(predict_list,episode_count):
    coin_fund = 10
    first_usdt = 1000
    usdt_fund = 120
    COIN_INDEX =2#XRPUSDT
    sav_count = 0
    for e in range(episode_count + 1):
        data_gen = generate_arrays_from_file_for_training(data_dir_name,LOOK_BACK,COIN_INDEX)
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
            reward = 0
            for i , coin in enumerate(predict_list):
                if action == 1 and usdt_fund > 0: # buy coin i-th
                    # print("price=",price)
                    num_coin = 0.999*coin_fund/price[i]
                    agent.inventory[coin].append([num_coin,price[i]])
                    print ("Buy: {} of {} at price {} ".format(coin,num_coin,price[i]))
                    usdt_fund -= coin_fund
                    reward -= 0.001*coin_fund
                elif action == 2 and len(agent.inventory[coin]) > 0: # sell
                    print("agent.inventory[{}]={}".format(coin,agent.inventory[coin]))
                    num_coin , bought_price = agent.inventory[coin].pop(0)   
                    coin_fund  = num_coin*price[i]*0.999     
                    usdt_fund += coin_fund             
                    reward += (price[i] - bought_price)*num_coin*0.999
                    total_profit += num_coin*(price[i] - bought_price)
                    print ("Sell: {} | Profit: {}".format(coin,num_coin*(price[i] - bought_price)))
                    # print ("Buy: {} of {} at price {} ".format(coin,num_coin,price[i]))
            total_coin= 0
            for [amount,bought_price] in agent.inventory[coin]:
                total_coin += amount
            equi_acc = total_coin*price[i] + usdt_fund
            print("reward=",reward)
            print("usdt_fund=",usdt_fund)
            prCyan("total_profit:{}".format(total_profit))
            prGreen("account value = {}".format(equi_acc ))
            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            # agent.memory.append((state, action, reward, next_state, done))

            state = next_state
            if done:
                print ("--------------------------------")
                print ("Total Profit: {}".format(total_profit))
                print ("--------------------------------")
            print("memory length =",len(agent.memory) )
           # if t%100 ==0:
                    #agent.model.save("model/model_tmp")

            if len(agent.memory) %(train_steps) == 0 and len(agent.memory) > batch_size:
                print("back propagation")
                agent.expReplay(batch_size)
                sav_count += 1
                if sav_count == 10:
                    sav_count = 0
                    agent.model.save("model/model_tmp")

        if e % 10 == 0:
            agent.model.save("model/model_ep" + str(e))
def csv_preprocessing(coin_list,path):
    size=[]
    for coin in coin_list:
        df = pd.read_csv(path  + "{}.csv".format(coin))
        size.append(df.shape[0])
    size_min = min(size)
    l = size_min-100
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

    allcoin_f= pd.DataFrame(index=None,columns=cols_name)
    for coin in coin_list:
        df = pd.read_csv(path  + "{}.csv".format(coin))
        allcoin_f["{}_High".format(coin)] = df['High']
        allcoin_f["{}_Low".format(coin)] = df['Low']
        allcoin_f["{}_Close".format(coin)] = df['Close']
        allcoin_f["{}_Volume".format(coin)] = df['Volume']
        allcoin_f["{}_Count".format(coin)] = df['Number of trades']
            
    allcoin_f.to_csv(path  + "allcoin.csv")                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decide to Train or to Run.')
    parser.add_argument('--mode', type=str, help='--mode=train or --mode=trainor --mode=predict')
    parser.add_argument('--coin', type=str, help='--coin=BTCUSDT')
    parser.add_argument('--ep', type=int, help='--ep=10')
    # parser.add_argument('--pw', type=str, help='--coin=BTCUSDT')
    args = parser.parse_args()
    if args.ep:
        episode_count = args.ep
    else:
        episode_count = 10
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
    if args.mode == "train":   
        predict_list='XRPUSDT'.split()
        agent = Agent([LOOK_BACK,DF_FEATURES,1],NUM_PREDICT_COIN*3,is_eval=False,model_name="model_tmp")
    
        deepq_train(predict_list,episode_count)

    elif args.mode=="processdata":
        csv_preprocessing(total_list,data_dir_name)
        pd_join_csv(total_list,data_dir_name)
    
        