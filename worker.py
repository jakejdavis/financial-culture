import names
import uuid
import numpy as np

from activity_model import ActivityModel

AMOUNT = 100

class Worker:

    def __init__(self, pool, weights = None, surname = None):
        self.name = names.get_first_name() + " " + surname if surname != None else names.get_full_name()
        self.pool = pool
        self.model = ActivityModel(weights)
        self.funds = 100000

        self.portfolio = []

        #print("[Pool %s][%s] Worker initialised" % (self.pool, self.name))

    def action(self, prediction_values_X, actual_values_X, tag):
        #print("[Worker %s] Acting..." % (self.name)) #Buy
        predict = self.model.predict(prediction_values_X)
        action = np.argmax(predict) # What value had the highest confidence

        success = False
        if action == 0:
            #print("[Worker %s] No Action" % (self.name)) #Nothing
            success = True
        elif action == 1:
            #print("[Worker %s] Buy at %f" % (self.name, actual_values_X[0][-1])) #Buy
            if self.funds >= AMOUNT * actual_values_X[0][-1]:
                self.portfolio.append({
                    "id": uuid.uuid4(),
                    "tag": tag,
                    "amount": AMOUNT,
                    "at_value": actual_values_X[0][-1]
                })
                success = True
            #else:
                #print("[Worker %s] Is a poor boye and can't afford to buy %s" % (self.name, tag))
        elif action == 2:
            #print("[Worker %s] Sell" % (self.name)) #Sell

            removed = False
            for stock in self.portfolio:
                if stock["tag"] == tag:
                    self.portfolio.remove(stock) 
                    removed = stock
                    break
                    
            if removed:
                funcs_inc = stock["amount"] * actual_values_X[0][-1]
                self.funds += funcs_inc
                bought_for = stock["amount"] * stock["at_value"]
                difference = funcs_inc - bought_for
                profit_or_loss = "profit" if difference > 0 else "loss"
                #print("[Worker %s] Sold %s for %f, a %s of %f" % (self.name, tag, funcs_inc, profit_or_loss, difference))
                success = True
            #else:
                #print("[Worker %s] Tried to sell %s, but doesn't have any %s stocks in its portfolio, wot" % (self.name, tag, tag)) 

        return action, success

    def get_funds(self):
        return self.funds