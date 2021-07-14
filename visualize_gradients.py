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
from classifiers.lstm import LSTM, LSTMGradientVisualizer
from classifiers.feed_forward import FeedForward, FeedForwardGradientVisualizer
from dubins_path_planner.scene import Scene

def build_model(modelSelection, inputShape):

    print('input shape:', inputShape)

    if modelSelection.lower() == 'feedforward':

        model = FeedForward(inputShape)

    elif modelSelection.lower() == 'lstm':

        model = LSTM()
        
    return model

def get_visualizer(modelSelection, dataset, model, weightsDir, scene, display):

    if modelSelection.lower() == 'lstm':

        visualizer = LSTMGradientVisualizer(model, dataset, weightsDir=weightsDir, scene=scene, display=display)

    return visualizer

def visualize_gradients(modelSelection, dataDirectory, algo='optimal_rrt', sceneName = 'test_room', display=True):

    loader = None

    valDataDir = os.path.join(dataDirectory, '{}_batches_validate'.format(algo))

    numBatchesToLoad = 1
    loader = ValidateDataLoader(numBatchesToLoad, valDataDir, stepSize=10)
    dataset = loader.load()

    inputShape = (1,3)
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

    args = parser.parse_args()

    modelSelection = args.model
    algorithm = args.algo.lower()
    dataDirectory = args.directory
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\'

    visualize_gradients(modelSelection, dataDirectory, algorithm)


