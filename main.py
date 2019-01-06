import tensorflow as tf
import keras
import pickle
import argparse
import random
import numpy as np

from pool import Pool

parser = argparse.ArgumentParser()

parser.add_argument("-gen", "--generate", help="Generate Workers")
parser.add_argument("-sav", "--save", help="Save Directory")
parser.add_argument("-rt", "--retrain", help="Retrain Models")
parser.add_argument("-plt", "--plot", help="Plot Graphs")
parser.add_argument("-eval", "--evaluate", help="Test Pool Models")
parser.add_argument("-sim", "--simulate", help="Simulate Activity Models")

pool_names = ["Vanguard", "Chase", "Barclay", "Deutsche"]
pools = []

tags = ["EOD/DIS", "EOD/HD", "EOD/MSFT", "EOD/BA", "EOD/MMM", 
"EOD/PFE", "EOD/NKE", "EOD/JNJ", "EOD/MCD", "EOD/INTC", 
"EOD/V", "EOD/IBM", "EOD/GE", "EOD/KO", "EOD/WMT", "EOD/AAPL"]

def main(): 
    global pools

    args = parser.parse_args()

    # Generate pools and workers
    to_train = []
    if args.generate == "yes":
        print("[Pool GEN] Generating pools...")
        for name in pool_names:
            pools.append(Pool(name))
        print("[Pool GEN] %d pools generated" % (len(pools)))
        for pool in pools:
            pool.generate_population()

    else:
        print("[Pool LOAD] Loading pools...")

        with open(args.save,'rb') as f:
            pools = pickle.load(f)

        print("[Pool LOAD] %d pools loaded" % (len(pools)))

    for pool in pools:
        if not pool.load_model():
            to_train.append(pool.pool_name)

    if args.retrain == "yes":
        to_train = pool_names

    print("[Pool TRAIN] Training %d unloaded models" % (len(to_train)))
    random.shuffle(to_train)
    for training_pool in to_train:
        pool = pools[pool_names.index(training_pool)]
        print("[Pool %s] Train start with diff activation of %d" % (pool.pool_name, pool.diff_activation))
        random.shuffle(tags)
        for tag in tags:
            print("[Pool %s] Fit on tag %s" % (pool.pool_name, tag))
            pool.train(tag)
            
            pool.save_model()
    print("[Pool TRAIN] Unloaded model train finished")
    
    if args.evaluate == "yes":
        print("[Pool EVAL] Evaluating models")
        for pool in pools:
            for tag in tags:
                predict = pool.predict(tag, True)

                correct = 0
                incorrect = 0
                total_difference = 0
                for i, prediction_X in enumerate(predict[1]):
                    actual_X = predict[0][i]
                    prediction = prediction_X[-1]
                    actual = actual_X[-1]
                    difference = abs(prediction-actual)
                    total_difference += difference

                    previous = actual_X[-2]

                    change = ""
                    if actual > previous:
                        change = "inc"
                    elif actual < previous:
                        change = "dec"
                    else:
                        change = "eq"
                    
                    predicted_change = ""
                    if prediction > previous:
                        predicted_change = "inc"
                    elif prediction < previous:
                        predicted_change = "dec"
                    else:
                        predicted_change = "eq"

                    if change == predicted_change:
                        correct += 1
                    else:
                        incorrect += 1
                print("[Pool EVAL] Pool model %s had %d correct and %d incorrect change predictions, total difference of %f" % (pool.pool_name, correct, incorrect, total_difference))


    if args.simulate == "yes":
        print("[Pool SIM] Starting simulations")

        for tag in tags:
            print("[Pool SIM] Starting simulation on tag %s" % tag)
            for pool in pools:
                pool.simulate(tag)
            
        
            for pool in pools:
                total_profit = pool.get_total_profit()
                print("[Pool SIM] Pool %s has made total profits of %f" % (pool.pool_name, total_profit))

if __name__ == "__main__":
    main()