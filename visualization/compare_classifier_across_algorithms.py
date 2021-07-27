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
from dubins_path_planner.scene import Scene

def label_plots(path_axes, preds_axes):

    optimal_path_ax, adversarial_path_ax = path_axes
    optimal_preds_ax, adversarial_preds_ax = preds_axes

    optimal_path_ax.set_title('Optimal RRT')
    adversarial_path_ax.set_title('Adversarial Optimal RRT')

    optimal_preds_ax.set_title('Confidence over time')
    adversarial_preds_ax.set_title('Confidence over time')

    return (optimal_path_ax, adversarial_path_ax), (optimal_preds_ax, adversarial_preds_ax)

def plot_preds(ax, preds):

    targetColors = ['blue','orange', 'green', 'pink', 'red']

    for targetIdx in range(len(targetColors)):

        conf = []

        for t in range(preds.shape[0]):

            conf.insert(0, preds[t, targetIdx])
        
        ax.plot(range(preds.shape[0], 0, -1), conf, linestyle='--', color=targetColors[targetIdx])

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

def plot_paths_and_preds(paths, preds, scene):

    fig = plt.figure()
    gs = gridspec.GridSpec(2,2, width_ratios=(1,1), height_ratios=(4,1))

    optimal_scene_ax = plt.subplot(gs[0,0])
    optimal_pred_ax = plt.subplot(gs[1,0])
    adversarial_scene_ax = plt.subplot(gs[0,1])
    adversarial_pred_ax = plt.subplot(gs[1,1])

    targetColors = ['blue','orange', 'green', 'pink', 'red']
    optimal_scene_ax = scene.draw(optimal_scene_ax, targetColors) 
    adversarial_scene_ax = scene.draw(adversarial_scene_ax, targetColors) 

    optimalPath, adversarialPath = paths

    optimal_scene_ax = plot_path(optimal_scene_ax, optimalPath['path'])
    adversarial_scene_ax = plot_path(adversarial_scene_ax, adversarialPath['path'])

    optimalPreds, adversarialPreds = preds

    time = max(optimalPreds.shape[0], adversarialPreds.shape[0])
    optimal_pred_ax.set_xlim(time, 0)
    optimal_pred_ax.grid(True)
    adversarial_pred_ax.set_xlim(time, 0)
    adversarial_pred_ax.grid(True)

    optimal_pred_ax = plot_preds(optimal_pred_ax, optimalPreds)
    adversarial_pred_ax = plot_preds(adversarial_pred_ax, adversarialPreds)

    path_axes = optimal_scene_ax, adversarial_scene_ax
    preds_axes = optimal_pred_ax, adversarial_pred_ax
    
    path_axes, preds_axes = label_plots(path_axes, preds_axes)
    
    plt.show()
    
def get_preds(paths, analyzer):

    optimalPath, adversarialPath = paths

    downSample = 100
    instance = np.array([optimalPath['path']['x'], optimalPath['path']['y'], optimalPath['path']['theta']]).transpose()
    instance[:, :2] /= 10
    instance[:, 2] -= math.pi
    instance[:, 2] /= math.pi
    instance = instance[np.newaxis, ::downSample, :] 
    print(instance)

    label = np.zeros((1,5))
    label[0, optimalPath['target']['index']] = 1.0

    _, optimalPreds= analyzer.analyze(instance, label)

    instance = np.array([adversarialPath['path']['x'], adversarialPath['path']['y'], adversarialPath['path']['theta']]).transpose()
    instance[:, :2] /= 10
    instance[:, 2] -= math.pi
    instance[:, 2] /= math.pi
    instance = instance[np.newaxis, ::downSample, :]
    print(instance)

    label = np.zeros((1,5))
    label[0, optimalPath['target']['index']] = 1.0

    _, adversarialPreds = analyzer.analyze(instance, label)

    return optimalPreds, adversarialPreds

def load_paths(pathNum):

    with open('../data/paths/optimal_rrt_path_{}.json'.format(pathNum), 'r') as f:
        optimalPath = json.load(f)

    with open('../data/paths/adversarial_optimal_rrt_path_{}.json'.format(pathNum), 'r') as f:
        adversarialPath = json.load(f)

    return optimalPath, adversarialPath

def get_analyzer(sceneName):

    #weightsDir = 'D:\\path_planning_data\\{}_dataset\\optimal_rrt_lstm_weights\\'.format(sceneName)
    weightsDir = '..\\data\\{}_dataset\\optimal_rrt_lstm_weights\\'.format(sceneName)
    model = LSTM()

    return LSTMGradientAnalyzer(model, weightsDir)

def compare_predictions(modelSelection, dataDirectory, algo='optimal_rrt', pathNum=0, sceneName = 'tower_defense'):

    analyzer = get_analyzer(sceneName)

    paths = load_paths(pathNum)
    preds = get_preds(paths, analyzer)

    scene = Scene(sceneName)
    plot_paths_and_preds(paths, preds, scene)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--directory', type = str, default = '..\\data\\')
    parser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "RRT")
    parser.add_argument('--path_num', type=int, default=0)

    args = parser.parse_args()

    modelSelection = args.model
    algorithm = args.algo.lower()
    dataDirectory = args.directory
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\'
    pathNum = args.path_num

    compare_predictions(modelSelection, dataDirectory, algorithm, pathNum)
