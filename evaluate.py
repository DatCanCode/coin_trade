import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys
import coin_pred
if len(sys.argv) != 3:
    print "Usage: python evaluate.py [stock] [model]"
    exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getcoinData(stock_name)
l = data.shape[0] - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range(l):
    action = agent.act(state)

    # sit
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0

    if action == 1: # buy
        agent.inventory.append(data[t,3])
        print "Buy: " + formatPrice(data[t,3])

    elif action == 2 and len(agent.inventory) > 0: # sell
        bought_price = agent.inventory.pop(0)
        reward = max(data[t,3] - bought_price, 0)
        total_profit += data[t,3] - bought_price
        print "Sell: " + formatPrice(data[t,3]) + " | Profit: " + formatPrice(data[t,3] - bought_price)

    done = True if t == l - 1 else False
    # agent.memory.append((state, action, reward, next_state, done))
    state = next_state
