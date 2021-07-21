import sys
import os
import argparse
import csv
import json
import numpy as np
import random

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import dubins_path_planner.planning_algorithms.RRT
from dubins_path_planner.run_RRT import run_RRT
from dubins_path_planner.run_optimal_RRT import run_optimal_RRT

def save_json_format(samplesInBatch, batchNum, sceneName, algo):

    with open('./{}_batches_new/batch_{}.json'.format(algo.lower(), batchNum), 'w') as outFile:
        json.dump(samplesInBatch, outFile)

def save_batch(samplesInBatch, batchNum, sceneName, algo):
    print('saving batch {}'.format(batchNum), flush=True)
    print('saving json')
    save_json_format(samplesInBatch, batchNum, sceneName, algo)

parser = argparse.ArgumentParser(description='Create batches of training data from RRT dubins planner.')

parser.add_argument('--batches', type=int, help='Number of batches to create.', default=1)
parser.add_argument('--batchsize', type=int, help='Number of paths generated per batch.', default=10)
parser.add_argument('--scene', type=str, help='Name of scene.', default='simple_room')
parser.add_argument('--format', type=str, help='Format of save file..', default='json')
parser.add_argument('--start_index', type=int, help='Index to begin saving batches at', default=0)
parser.add_argument('--algo', type=str, help='RRT or optimal', default = 'RRT')

args = parser.parse_args()

for batchNum in range(args.start_index, args.start_index + args.batches):

    samplesInBatch = {}
    for sampleNum in range(args.batchsize):

        print('sample number: {}'.format(sampleNum), flush=True)

        # uncomment for debugging
        """
        seed = random.randint(1, 100000)
        print('seed:', seed)
        np.random.seed(seed)
        random.seed(seed)
        """

        sample = None 
        while sample is None:

            print('running RRT*')
            if args.algo == 'RRT':
                sample = run_RRT(animate=False, sceneName=args.scene)
            else:
                try:
                    sample = run_optimal_RRT(animate=False, sceneName=args.scene)
                except Exception as e:
                    print('an exception occurred')
                    print(e)
                    sample = None

        samplesInBatch['{}'.format(sampleNum)] = sample

    save_batch(samplesInBatch, batchNum, args.scene, args.algo)

print('finished')



        

