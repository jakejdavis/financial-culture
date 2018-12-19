from keras.layers.core import Dense, Activation, Dropout, Flatten   
from keras.layers.recurrent import GRU
from keras.layers import TimeDistributed, Embedding
from keras import optimizers

from sklearn.model_selection import train_test_split
import numpy as np

from model import Model
from interpret import values_from_tag

LOSS = "mean_squared_error"
OPTIMIZER = "adam"

in_out_neurons = 2  
hidden_neurons = 2

class PredictModel(Model):
    fitness = 0

    def create(self):
        
        input_shape = values_from_tag("EOD/DIS")[0][0].shape
        self.model.add(GRU(3, input_shape=input_shape, return_sequences=True))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='linear'))

        self.model.summary()
        self.compile()

    def compile(self):
        self.model.compile(loss=LOSS, optimizer=OPTIMIZER)

    def predict(self, values_X):
        return self.model.predict(np.array(values_X))

