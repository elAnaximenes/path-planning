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
import math
from matplotlib import gridspec

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from data.loaders.val_loader import ValidateDataLoader
from data.loaders.mean_path_loader import MeanPathDataLoader 
from classifiers.lstm import LSTM, LSTMGradientAnalyzer
from classifiers.feed_forward import FeedForward, FeedForwardGradientAnalyzer
from dubins_path_planner.scene import Scene

def label_plots(path_ax, preds_ax):

    path_ax.set_title('Path and Confidence')

    return path_ax, preds_ax

def plot_preds(ax, preds, targetColors):

    for targetIdx in range(len(targetColors)):

        conf = []

        for t in range(preds.shape[0]):

            conf.append(preds[t, targetIdx])
        
        ax.plot(range(0, preds.shape[0]), conf, linestyle='--', color=targetColors[targetIdx])

    return ax

def get_path(path):

    x = []
    y = []

    for t in range(len(path['x'])):

        x.append(path['x'][t])
        y.append(path['y'][t])

    return x, y

def plot_path(ax, path):

    x, y = get_path(path)
    ax.plot(x, y, 'b--')

    return ax

def plot_paths_and_preds(path, preds, scene):

    fig = plt.figure()
    gs = gridspec.GridSpec(2,1, height_ratios=(4,1))
    scene_ax = plt.subplot(gs[0,0])
    pred_ax = plt.subplot(gs[1,0])

    fig.tight_layout(pad=3.0)

    targetColors = ['blue','orange', 'green', 'pink', 'red']
    print(targetColors)
    scene_ax = scene.draw(scene_ax, targetColors) 

    scene_ax = plot_path(scene_ax, path['path'])

    pred_ax = plot_preds(pred_ax, preds, targetColors)
    
    scene_ax, preds_ax = label_plots(scene_ax, pred_ax)
    
    plt.show()
    
def get_pred(path, analyzer):

    instance = np.array([path['path']['x'], path['path']['y'], path['path']['theta']])

    downSample = 100

    instance = instance.transpose()
    instance[:, :2] /= 10.0
    instance[:, 2] -= math.pi
    instance[:, 2] /= math.pi
    instance = instance[np.newaxis, ::downSample, :]

    label = np.zeros((1,5))
    label[0, path['target']['index']] = 1.0

    _, preds= analyzer.analyze(instance, label)

    return preds 

def load_path(pathNum):

    with open('../data/paths/optimal_rrt_path_{}.json'.format(pathNum), 'r') as f:
        path = json.load(f)

    return path 

def get_analyzer(sceneName, modelSelection):

    analyzer = None
    if modelSelection == 'lstm':
        weightsDir = '..\\data\\{}_dataset\\optimal_rrt_lstm_weights_predictor\\'.format(sceneName)
        model = LSTM()
        analyzer = LSTMGradientAnalyzer(model, weightsDir)
    elif modelSelection == 'feedforward':
        weightsDir = '..\\data\\{}_dataset\\optimal_rrt_feedforward_weights\\'.format(sceneName)
        analyzer = FeedForwardGradientAnalyzer(weightsDir)

    return analyzer

def compare_predictions(modelSelection, dataDirectory, algo='optimal_rrt', pathNum=0, sceneName = 'tower_defense'):

    analyzer = get_analyzer(sceneName, modelSelection)

    path = load_path(pathNum)
    pred = get_pred(path, analyzer)

    scene = Scene(sceneName)
    plot_paths_and_preds(path, pred, scene)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "LSTM")
    parser.add_argument('--directory', type = str, default = '..\\data\\')
    parser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "optimal_rrt")
    parser.add_argument('--path_num', type=int, default=0)

    args = parser.parse_args()

    modelSelection = args.model.lower()
    algorithm = args.algo.lower()
    dataDirectory = args.directory
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\'
    pathNum = args.path_num

    compare_predictions(modelSelection, dataDirectory, algorithm, pathNum)
