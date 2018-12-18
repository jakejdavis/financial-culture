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

pool_names = ["Vanguard", "Chase", "Barclay", "Deutsche"]
pools = []

tags = ["EOD/DIS", "EOD/HD", "EOD/MSFT"]

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
        print(pools)
        for pool in pools:
            pool.generate_population()

        with open(args.save,'wb') as f:
            pickle.dump(pools, f)
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
    for training_pool in to_train:
        pool = pools[pool_names.index(training_pool)]
        print("[Pool %s] Train start with diff activation of %d" % (pool.pool_name, pool.diff_activation))
        random.shuffle(tags)
        for tag in tags:
            print("[Pool %s] Fit on tag %s" % (pool.pool_name, tag))
            pool.train(tag)
            
            pool.save_model()
    print("[Pool TRAIN] Unloaded model train finished")
    
            
            


    ## Repeat 

        ## Interpret data

        ## Fit pool models

        ## Get prediction

        ## Pass prediction to workers

        ## Check workers predictions
        # Penalise & reward large predictions greater then small predictions

        ## Create new workers from fitness
        

    

if __name__ == "__main__":
    main()