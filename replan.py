import tensorflow as tf
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from data.val_loader import ValidateDataLoader
from classifiers.lstm import LSTM, LSTMGradientVisualizer
from classifiers.feed_forward import FeedForward, FeedForwardGradientVisualizer
from dubins_path_planner.RRT import Scene
import visualize_gradients

def get_visualizer(modelSelection, dataDirectory, algorithm):

    return visualize_gradients.visualize_gradients(modelSelection, dataDirectory, algorithm, display=False)

def replan(modelSelection, dataDirectory, algorithm):

    visualizer = get_visualizer(modelSelection, dataDirectory, algorithm)
    visualizer.visualize_single_instance()
    plt.show()

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

    replan(modelSelection, dataDirectory, algorithm)
