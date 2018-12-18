import quandl
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.models import load_model

from worker import Worker
from predict_model import PredictModel
from interpret import values_from_tag

POOL_SIZE = 50


class BatchOutput(Callback):
    def __init__(self, pool_name):
        self.pool_name = pool_name
    def on_batch_end(self, batch, logs={}):
        print("[Pool %s] Batch: %d\tLoss: %d" % (self.pool_name, 
            logs.get("batch"), logs.get('loss')))

class Pool:
    def __init__(self, pool_name):
        self.diff_activation = random.uniform(-1, 1)
        self.pool_name = pool_name
        self.population = []
        self.predict_model = PredictModel()
        
    def generate_population(self):
        print("[Pool %s][GEN] Generation starting..." % (self.pool_name))
        for i in range(POOL_SIZE):
            #print("[Pool %s][GEN] Generating worker %d" % (self.pool_name, i))
            self.population.append(Worker(self.pool_name))
        print("[Pool %s][GEN] Generated population of %d" % (self.pool_name, len(self.population)))
 
    def train(self, tag):
        self.predict_model.compile()

        self.batch_output_callback = BatchOutput(self.pool_name)

        values_X, values_Y = values_from_tag(tag)


        train_X, test_X, train_Y, test_Y = train_test_split(values_X, values_Y, test_size=0.33)
        print(train_X.shape)
        return self.predict_model.model.fit(
            train_X, train_Y,
            validation_data=(test_X, test_Y), 
            epochs=20, batch_size=5, verbose=0, callbacks=[self.batch_output_callback])

    def train_output(self, history):
        print(history)
        print("[Pool %s] Epoch: %d Loss: %d, Accuracy: %d" % (self.pool_name, history["epoch"], history["loss"], history["acc"]))


    def save_model(self):
        print("[Pool %s] Saving pool model..." % (self.pool_name))
        self.predict_model.model.save('models/%s.h5' % (self.pool_name))

    def load_model(self):
        print("[Pool %s] Loading pool model..." % (self.pool_name))
        try:
            self.predict_model.model = load_model('models/%s.h5' % (self.pool_name))
            print("[Pool %s] Pool model loaded" % (self.pool_name))
            return True
        except:
            print("[Pool %s] Pool model load failure" % (self.pool_name))

    def predict(self, tag):
        values_X, values_Y = values_from_tag(tag)

        prediction = self.predict_model.predict(values_X)

        return values_X, values_Y, prediction

        
