import tensorflow as tf
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
from data.val_loader import ValidateDataLoader
from classifiers.lstm import LSTM, LSTMGradientAnalyzer
from dubins_path_planner.RRT import Scene
import visualize_gradients

class AdversarialPlanner:

    def __init__(self, perturbationRate=0.001):

        self.perturbationRate = perturbationRate
        pass

    def add_noise(self, grads, instance):

        #instance is (1, timesteps, features)

        noisyPath = instance
        
        noisyPath[0] += grads*self.perturbationRate

        return noisyPath

def is_fooled(predictions, label):

    #for pred in predictions:

        #if np.argmax

    return False 

def get_path(path):

    x = []
    y = []

    for t in range(path.shape[1]):

        x.append(path[0,t, 0]*10)
        y.append(path[0,t, 1]*10)

    return x, y

def plot_paths_and_predictions(orig, perturbed, scene, it):
    
    
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(4,2, width_ratios=(1,1))
    orig_ax = plt.subplot(gs[:3, 0])
    orig_pred_ax = plt.subplot(gs[3, 0])
    perturbed_ax = plt.subplot(gs[:3, 1])
    perturbed_pred_ax = plt.subplot(gs[3, 1])

    orig_ax.set_xlim(scene.dimensions['xmin'], scene.dimensions['xmax'])
    perturbed_ax.set_xlim(scene.dimensions['xmin'], scene.dimensions['xmax'])
    orig_ax.set_aspect('equal')
    perturbed_ax.set_aspect('equal')

    targetColors = ['green', 'blue', 'cyan', 'darkorange', 'purple']
    for obstacle in scene.obstacles:

        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        orig_ax.add_patch(obs)
        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        perturbed_ax.add_patch(obs)

    for i, target in enumerate(scene.targets):

        tar = plt.Circle((target[0], target[1]), target[2], color=targetColors[i], fill=False)
        orig_ax.add_patch(tar)
        tar = plt.Circle((target[0], target[1]), target[2], color=targetColors[i], fill=False)
        perturbed_ax.add_patch(tar)

    origInstance, origPredictions = orig
    perturbedPath, perturbedPredictions = perturbed

    orig_x, orig_y = get_path(origInstance)
    pert_x, pert_y = get_path(perturbedPath)

    orig_ax.scatter(orig_x, orig_y)
    perturbed_ax.scatter(pert_x, pert_y)

    timeSteps = range(0, len(origPredictions), 10)

    orig_pred_ax.set_ylim(-0.05, 1.05) 
    perturbed_pred_ax.set_ylim(-0.05, 1.05) 


    for targetIdx in range(len(targetColors)):

        origConf = []
        pertConf = []

        for t in range(len(timeSteps)):

            origConf.append(origPredictions[t][0][targetIdx])
            pertConf.append(perturbedPredictions[t][0][targetIdx])

        orig_pred_ax.plot(timeSteps, origConf, linestyle='--', color=targetColors[targetIdx])
        perturbed_pred_ax.plot(timeSteps, pertConf, linestyle='--', color=targetColors[targetIdx])

    plt.savefig('./data/saved_images/img-{}'.format(it))
    plt.show()

def get_dataset(modelSelection, dataDirectory, algorithm):

    dataset = None 
    
    valDataDir = os.path.join(dataDirectory, '{}_batches_validate'.format(algorithm)) 
    if modelSelection.lower() == 'lstm':
        loader = ValidateDataLoader(numBatches=1, dataDirectory=valDataDir, stepSize = 10)

    dataset = loader.load()

    return dataset

def get_gradient_analyzer(modelSelection, dataDirectory, algorithm):

    weightsDir = os.path.join(dataDirectory, '{}_{}_weights'.format(algorithm, modelSelection.lower()))
    analyzer = None

    if modelSelection.lower() == 'lstm':
        model = LSTM()
        analyzer = LSTMGradientAnalyzer(model, weightsDir)

    return analyzer

def replan(modelSelection, dataDirectory, algorithm):

    analyzer = get_gradient_analyzer(modelSelection, dataDirectory, algorithm)
    dataset = get_dataset(modelSelection, dataDirectory, algorithm)
    planner = AdversarialPlanner()
    scene = Scene('test_room')

    for instance, label in dataset.data:

        fooled = False
        # transforms instance into timeseries
        grads, originalPredictions, instance = analyzer.analyze(instance, label)
        origInstance = np.copy(instance)

        it = 0
        while not fooled:

            noisyPath = planner.add_noise(grads, instance)
            grads, noisyPredictions, instance = analyzer.analyze(noisyPath, label)

            plot_paths_and_predictions((origInstance, originalPredictions), (noisyPath, noisyPredictions), scene, it)

            fooled = is_fooled(noisyPredictions, label)
            it += 1

        #spaPath, spaPredictions = planner.single_point_attack(grads, instance)
        #instance = origInstance

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
