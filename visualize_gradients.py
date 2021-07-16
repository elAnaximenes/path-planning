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
from data.val_loader import ValidateDataLoader
from data.mean_path_loader import MeanPathDataLoader 
from classifiers.lstm import LSTM, LSTMGradientVisualizer
from classifiers.feed_forward import FeedForward, FeedForwardGradientVisualizer
from dubins_path_planner.scene import Scene

def build_model(modelSelection, inputShape):

    if modelSelection.lower() == 'feedforward':
        model = FeedForward(inputShape)
    elif modelSelection.lower() == 'lstm':
        model = LSTM()
        
    return model

def get_visualizer(modelSelection, dataset, model, weightsDir, scene, display):

    if modelSelection.lower() == 'lstm':
        visualizer = LSTMGradientVisualizer(model, dataset, weightsDir=weightsDir, scene=scene, display=display)
    if mdelSelection.lower() == 'feedforward':
        visualizer = FeedForwardGradientVisualizer(model, dataset, weightsDir=weightsDir, scene=scene, display=display)

    return visualizer

def get_data_loader(dataDir, algo, meanPaths):

    loader = None

    if meanPaths:
        valDataDir = os.path.join(dataDirectory, '{}_batches_train'.format(algo))
        numBatchesToLoad = 10
        loader = MeanPathDataLoader(numBatchesToLoad, valDataDir, loadMeanPaths=True)
    else:
        valDataDir = os.path.join(dataDirectory, '{}_batches_validate'.format(algo))
        numBatchesToLoad = 1
        loader = ValidateDataLoader(numBatchesToLoad, valDataDir, stepSize=10)

    return loader

def visualize_gradients(modelSelection, dataDirectory, algo='optimal_rrt', sceneName = 'test_room', display=True, meanPaths=False):

    loader = get_data_loader(dataDirectory, algo, meanPaths)
    dataset = loader.load()

    inputShape = (3,1000)
    model = build_model(modelSelection, inputShape)

    weightsDir = os.path.join(dataDirectory, '{}_{}_weights'.format(algo, modelSelection.lower()))

    scene = Scene(sceneName)

    visualizer = get_visualizer(modelSelection, dataset, model, weightsDir, scene, display = display)

    if display:
        visualizer.visualize()

    return visualizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--directory', type = str, default = '.\\data\\')
    parser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "RRT")
    parser.add_argument('--mean_paths', default=False, action='store_true')

    args = parser.parse_args()

    modelSelection = args.model
    algorithm = args.algo.lower()
    dataDirectory = args.directory
    meanPaths = args.mean_paths
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\'

    visualize_gradients(modelSelection, dataDirectory, algorithm, meanPaths=meanPaths)


