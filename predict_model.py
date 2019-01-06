from keras.layers.core import Dense, Activation, Dropout, Flatten   
from keras.layers import Average
from keras.layers.recurrent import GRU, LSTM
from keras.layers import TimeDistributed, Embedding
from keras import optimizers

from sklearn.model_selection import train_test_split
import numpy as np

from model import Model
from interpret import values_from_tag

LOSS = "mse"
OPTIMIZER = "rmsprop"

class PredictModel(Model):
    def create(self, weights=None):
        # Input shape should always be 50, 1 lol

        self.model.add(LSTM(
            50, input_shape=(50, 1),
            return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(
            100,
            return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1))
        self.model.add(Activation('linear'))

        self.compile()

    def compile(self):
        self.model.compile(loss=LOSS, optimizer=OPTIMIZER)

