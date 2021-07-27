import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from data.loaders.val_loader import ValidateDataLoader
from lstm import LSTM, LSTMTester
from feed_forward import FeedForward, FeedForwardTester

def plot_performance(performance, modelSelection, algo):

    print(performance)
    tp = performance['tp']
    labelCount = performance['label count']

    accuracyOverTime = []
    correct = []
    pathsAlive = []
    timesteps = sorted(labelCount.keys())
    totalPaths = float(labelCount[timesteps[0]])

    for key in timesteps: 

        percent = float(tp[key])/float(labelCount[key])
        accuracyOverTime.append(percent)
        correct.append(float(tp[key])/totalPaths)
        pathsAlive.append(float(labelCount[key])/totalPaths)

    print(pathsAlive)
    print(correct)

    plt.title('{} Accuracy over time -- {}'.format(modelSelection, algo))
    plt.plot(timesteps, accuracyOverTime, 'b', label = 'Accuracy')
    plt.plot(timesteps, correct, 'r--', label = 'True Positives')
    plt.plot(timesteps, pathsAlive, 'g--', label = 'Path Count')
    plt.xlim(0, timesteps[-1])
    plt.ylim(0.0, 1.0)
    plt.xlabel('Timesteps from goal')
    plt.ylabel('% correct')
    plt.legend()
    plt.grid(True)
    plt.show()

def build_model(modelSelection, inputShape):

    print('input shape:', inputShape)

    if modelSelection.lower() == 'feedforward':

        model = FeedForward(inputShape)

    elif modelSelection.lower() == 'lstm':

        model = LSTM()
        
    return model

def get_tester(modelSelection, dataset, model, weightsDir):

    if modelSelection.lower() == 'feedforward':

        tester = FeedForwardTester(dataset, model, weightsDir)

    elif modelSelection.lower() == 'lstm':

        tester = LSTMTester(dataset, model, weightsDir)

    return tester

def test_model(modelSelection, dataDirectory, numBatches, algo='RRT', sceneName='tower_defense'):

    loader = None

    trainingDataDir = os.path.join(dataDirectory, '{}_dataset/{}_batches_validate'.format(sceneName, algo)) 
    loader =ValidateDataLoader(numBatches, trainingDataDir)
    dataset = loader.load()
    print('dataset loaded')

    inputShape = (3,)
    model = build_model(modelSelection, inputShape)
    print('model loaded')

    weightsDir = os.path.join(dataDirectory, '{}_dataset/optimal_rrt_{}_weights'.format(sceneName, modelSelection.lower()))
    print(weightsDir)
    tester = get_tester(modelSelection, dataset, model, weightsDir) 
    print('tester loaded')

    performance = tester.test()

    plot_performance(performance, modelSelection, algo)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Test selected model on validation or test set')
    argparser.add_argument('--model', type=str, default='FeedForward')
    argparser.add_argument('--directory', type=str, default='../data/')
    argparser.add_argument('--batches', type=int, default=1)
    argparser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "RRT")
    argparser.add_argument('--scene', type=str, help='Which scene to train over.', default = "tower_defense")

    args = argparser.parse_args()

    modelSelection = args.model
    dataDirectory = args.directory
    algorithm = args.algo.lower()
    sceneName = args.scene

    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data'
    numBatches = args.batches

    test_model(modelSelection, dataDirectory, numBatches, algorithm, sceneName)
