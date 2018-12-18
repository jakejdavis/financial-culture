import names

from activity_model import ActivityModel

class Worker:

    def __init__(self, pool, weights = None):
        self.name = names.get_full_name()
        self.pool = pool
        self.model = ActivityModel(weights)

        #print("[Pool %s][%s] Worker initialised" % (self.pool, self.name))