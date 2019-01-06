import quandl
import numpy as np
import random
from keras.models import load_model as keras_load_model

from worker import Worker
from predict_model import PredictModel
from interpret import values_from_tag, denormalise_windows

POOL_SIZE = 50


class Pool:
    def __init__(self, pool_name):
        self.diff_activation = random.uniform(-1, 1)
        self.pool_name = pool_name
        self.population = []
        self.predict_model = PredictModel()
        self.total_profit = 0

        self.retain = 0.2
        self.random_select = 0.04
        

    def generate_population(self):
        print("[Pool %s][GEN] Generation starting..." % (self.pool_name))
        for i in range(POOL_SIZE):
            #print("[Pool %s][GEN] Generating worker %d" % (self.pool_name, i))
            self.population.append(Worker(self.pool_name))
        print("[Pool %s][GEN] Generated population of %d" % (self.pool_name, len(self.population)))
 

    def train(self, tag):
        self.predict_model.compile()

        train_X, train_Y, test_X, test_Y = values_from_tag(tag, 50, True)

        print(train_X.shape)
        self.predict_model.model.fit(
            train_X, train_Y,
            validation_split=0.05, 
            epochs=1, batch_size=512)


    def train_output(self, history):
        print(history)
        print("[Pool %s] Epoch: %d Loss: %d, Accuracy: %d" % (self.pool_name, history["epoch"], history["loss"], history["acc"]))


    def save_model(self):
        print("[Pool %s] Saving pool model..." % (self.pool_name))
        self.predict_model.model.save('models/%s.h5' % (self.pool_name))


    def load_model(self):
        print("[Pool %s] Loading pool model..." % (self.pool_name))
        self.predict_model = PredictModel()
        
        try:
            self.predict_model.model.load_weights('models/%s.h5' % (self.pool_name))
            print("[Pool %s] Pool model loaded" % (self.pool_name))
            return True
        except Exception as e:
            print("[Pool %s] Pool model load failure:" % (self.pool_name))
            print("\t", e)


    def predict(self, tag, denormalise):
        train_X, train_Y, test_X, test_Y  = values_from_tag(tag, 50, True)

        print("[Pool %s] Prediction start" % (self.pool_name))
        actuals = test_Y
        predictions = self.predict_model.predict(test_X)

        print("[Pool %s] Denormalising predictions" % (self.pool_name))
        predictions_X = []
        actuals_X = []
        for i, prediction in enumerate(predictions):
            prediction_X = np.append(test_X[i], prediction)
            predictions_X.append(prediction_X)
        for i, actual in enumerate(actuals):
            actual_X = np.append(test_X[i], actual)      
            actuals_X.append(actual_X)
        
        predictions = denormalise_windows(np.array(predictions_X), tag)
        actuals = denormalise_windows(np.array(actuals_X), tag)

        return actuals, predictions


    def simulate(self, tag):
        predict = self.predict(tag, True)
        actions = {0: 0, 1: 0, 2: 0}
        print("[Pool %s][SIM] Starting worker acting..." % (self.pool_name))
        for x, prediction_X in enumerate(predict[1]):
            for i, worker in enumerate(self.population):
                worker_action = worker.action(np.array([prediction_X]), np.array([predict[0][x]]), tag)
                if worker_action[1]: actions[worker_action[0]] += 1
                
            if (x/len(predict[1])*100) % 5 == 0:
                print("[Pool %s][SIM] %d%% complete" % (self.pool_name, (x/len(predict[1])*100)))
        
        print("[Pool %s][SIM] For tag %s, %d bought, %d sold for %d windows" % (self.pool_name, tag, actions[1], actions[2], len(predict[1])))

        init_worker_funds = 100000 * len(self.population)
        worker_funds = self.get_worker_funds()
        print("[Pool %s][SIM] Total worker funds: %f" % (self.pool_name, worker_funds))
        difference = worker_funds - init_worker_funds
        self.total_profit += difference
        profit_or_loss = "profit" if difference > 0 else "loss"
        print("[Pool %s][SIM] Made a %f %s" % (self.pool_name, abs(difference), profit_or_loss))

        self.generate_new_population()


    def get_worker_funds(self):
        pool_total = 0
        for worker in self.population:
            pool_total += worker.get_funds()
        return pool_total


    def get_total_profit(self):
        return self.total_profit


    def generate_new_population(self):
        """
        Based off of https://gist.githubusercontent.com/harvitronix/d0a4a40cb32ead6c55d405631414ffeb/raw/9542f841686e223e2f8da850a5a6be722f9a22d0/evolve.py
        """
        print("[Pool %s][NEW_GEN] Generating new population" % (self.pool_name))
        old_population = []
        for worker in self.population:
            old_population.append({
                "worker": worker,
                "funds": worker.get_funds()
            })
        old_population = sorted(old_population, key=lambda w: w["funds"]) 

        retain_length = int(len(old_population)*self.retain)
        print("[Pool %s][NEW_GEN] Retain length of %d" % (self.pool_name, retain_length))
        parents = old_population[:retain_length]

        for worker in old_population[retain_length:]:
            if self.random_select > random.random():
                parents.append(worker)

        print("[Pool %s][NEW_GEN] Collected %d parents" % (self.pool_name, len(parents)))

        new_population = []
        while len(new_population) < POOL_SIZE:
            parent_x = random.choice(parents)["worker"]
            parent_y = random.choice(parents)["worker"]

            weights_x = []
            for layer in parent_x.model.model.layers:
                weights_x.append(layer.get_weights())
            weights_y = []
            for layer in parent_y.model.model.layers:
                weights_y.append(layer.get_weights())

            new_weights = []
            for i in range(len(weights_x)):
                new_weights.append(random.choice([weights_x[i], weights_y[i]]))

            surname = parent_x.name.split(" ")[1]

            new_population.append(Worker(self.pool_name, new_weights, surname))
        print("[Pool %s][NEW_GEN] Generated %d new workers" % (self.pool_name, len(new_population)))
        self.population = new_population

