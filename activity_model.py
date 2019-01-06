from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import GRU
from keras.layers import TimeDistributed
from keras import optimizers

from sklearn.model_selection import train_test_split
import numpy as np

OPTIMIZER = "adam"
LOSS = "binary_crossentropy"

from model import Model

class ActivityModel(Model):
    action_size = 3

    def create(self, weights=None):
        if weights != None:
            self.model.add(Dense(51, activation='relu', weights=weights[0]))
            self.model.add(Dense(24, activation='relu', weights=weights[1]))
            self.model.add(Dense(self.action_size, activation='sigmoid', weights=weights[2]))
        else:
            self.model.add(Dense(51, activation='relu', kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
            self.model.add(Dense(24, activation='relu', kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
            self.model.add(Dense(self.action_size, activation='sigmoid', kernel_initializer='random_uniform',
                    bias_initializer='zeros'))

        self.compile()

    def compile(self):
        self.model.compile(loss=LOSS, optimizer=OPTIMIZER)

    def mutate(self):
        weights = self.model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self.model.set_weights(weights)

    