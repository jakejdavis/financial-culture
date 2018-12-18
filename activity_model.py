from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import GRU
from keras.layers import TimeDistributed
from keras import optimizers

from sklearn.model_selection import train_test_split
import numpy as np

OPTIMIZER = optimizers.Adam(lr=0.1)
LOSS = "mse"

from model import Model

class ActivityModel(Model):
    state_size = 3
    action_size = 3
    
    def compile(self):
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss=LOSS, optimizer=OPTIMIZER)

    