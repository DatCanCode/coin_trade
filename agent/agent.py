import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

from keras.layers import Dense, Dropout, Activation ,Embedding
from keras.layers import LSTM
from keras.layers import Conv2D,Conv1D, MaxPooling2D,MaxPooling1D,Dropout, Flatten,GlobalAveragePooling1D
from keras.models import model_from_json
from keras import backend as K
from keras.utils import to_categorical

from resnet import resnet


import numpy as np
import random
from collections import deque

# DATA_INTERVAL_PER_STEP 5
# lstm_output_size = 6*3*64
COIN_FEATURE = 5
NUMCOINS=9
EXTRA_COINS=13
EXTRA_COIN_FEATURES=EXTRA_COINS*COIN_FEATURE
DF_FEATURES=COIN_FEATURE*NUMCOINS+EXTRA_COIN_FEATURES
#number of freq
 # Training

OUTPUT_SIZE=1*3
# epochs = 10
TIME_BETWEEN_SAMPLE = 30
seq_len=int((3*24*60)/TIME_BETWEEN_SAMPLE)
LAG_COEFF = 1
# input_shape = (LAG_COEFF,12,1)
LOOK_BACK=int(10*24*60.0/TIME_BETWEEN_SAMPLE)  #5min*24 = 120min = 2hrs
LOOK_BACK_IN_MS = LOOK_BACK*TIME_BETWEEN_SAMPLE*60*1000

data_dir_name  = "./data/kline/"
model_dir = "./model/"

class Agent:
    def __init__(self, state_size,numoutput, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = numoutput # sit, buy, sell
        self.memory = deque(maxlen=1024)
        self.inventory = {}
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = load_model("model/" + model_name) if is_eval else self._model(input_shape = [LOOK_BACK,DF_FEATURES,1],numoutput=OUTPUT_SIZE)

    def _model(self,input_shape ,numoutput):
        DIM_ORDERING = {'th', 'tf'}

        model = resnet.ResnetBuilder.build_resnet_101(input_shape, numoutput)
        print("build model with input shape {} ,output shape len{}".format(input_shape,numoutput))
        for ordering in DIM_ORDERING:
            K.set_image_dim_ordering(ordering)
            adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            # model.compile(loss="mse", optimizer=adam)
            model.compile(loss="mse", optimizer=adam, metrics=['mse'])

            assert True, "Failed to compile with '{}' dim ordering".format(ordering)
        return model

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return np.random.randint(3)#, size=self.action_size)

        options = self.model.predict(state)
        # options = np.array(options).reshape(-1,3)
        # return np.argmax(options,axis=1)
        return np.argmax(options)

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 
