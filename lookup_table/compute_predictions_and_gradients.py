import json
import csv
import argparse
import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import json
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from dubins_path_planner.scene import Scene
from data.loaders.val_loader import ValidateDataLoader
from classifiers.lstm import LSTM, LSTMGradientAnalyzer
from classifiers.feed_forward import FeedForward, FeedForwardGradientAnalyzer

def save_predictions(target, preds, modelSelection):
    
    fileName = './predictions/{}/preds_{}.csv'.format(modelSelection, target)
    with open(fileName, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for pred in preds:
            writer.writerow(pred.tolist())

def save_gradients(target, grads, modelSelection):

    fileName = './gradients/{}/grads_{}.csv'.format(modelSelection, target)
    with open(fileName, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for grad in grads:
            writer.writerow(grad.tolist())

def get_dataset(dataDir, algo, numBatches):

    valDataDir = os.path.join(dataDir, '{}_batches_train'.format(algo))
    stepSize = 100
    loader = ValidateDataLoader(numBatches, valDataDir, stepSize)
    dataset = loader.load()

    return dataset 

def get_gradient_analyzer(modelSelection, dataDirectory, algorithm):

    weightsDir = os.path.join(dataDirectory, '{}_{}_weights'.format(algorithm, modelSelection.lower()))
    analyzer = None

    if modelSelection.lower() == 'lstm':
        model = LSTM()
        analyzer = LSTMGradientAnalyzer(model, weightsDir)
        print('got analyzer')
    elif modelSelection.lower() == 'feedforward':
        model = FeedForward()
        analyzer = FeedForwardGradientAnalyzer(model, weightsDir)

    return analyzer

def get_model(modelSelection):

    if modelSelection == 'lstm':
        model = LSTM()

    return model

def compute_predictions_and_gradients(dataDir, algo, numBatches, modelSelection, target):

    sceneName = 'test_room'
    scene = Scene(sceneName)
    model = get_model(modelSelection)
    analyzer = get_gradient_analyzer(modelSelection, dataDir, algo)

    dataset = get_dataset(dataDir, algo, numBatches)

    predictions = []
    gradients = []

    i = 0
    for instance, label in dataset.data:

        if np.argmax(label) != target:
            continue
            
        grads, preds = analyzer.analyze(instance, label)
        predictions.append(preds)
        gradients.append(grads)
        print(i, flush=True)
        i += 1

    save_predictions(target, predictions)
    save_gradients(target, gradients)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--directory', type=str, default = './data/batches-validate')
    parser.add_argument('--algo', type=str, help='Planning algorithm', default='optimal_rrt')
    parser.add_argument('--batches', type=int, help='number of training batches to load', default=10)
    parser.add_argument('--model', type=str, default = 'lstm')
    parser.add_argument('--target', type=int, default = 0)

    args = parser.parse_args()

    dataDir = args.directory
    if dataDir == 'tower':
        dataDir = 'D:\\path_planning_data\\'

    algo = args.algo
    numBatches = args.batches
    modelSelection = args.model.lower()
    target = args.target

    compute_predictions_and_gradients(dataDir, algo, numBatches, modelSelection, target)
