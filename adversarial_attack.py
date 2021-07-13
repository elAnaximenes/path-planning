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

    def __init__(self, perturbationRate=0.1):

        self.perturbationRate = perturbationRate
        pass

    def add_noise(self, grads, instance, gtIdx, perturbIdx):

        #instance is (1, timesteps, features)

        noisyPath = instance
        noisyPath[0] += (grads[:,perturbIdx,:] - grads[:,gtIdx,:])*self.perturbationRate

        return noisyPath
    
    def one_pixel_attack(self, grads, instance, gtIdx, perturbIdx):

        attackedPath = instance

        print(grads.shape)

        #maxGradient = 


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

def plot_paths_and_predictions(paths, predictions, scene, it):
    
    fig = plt.figure(figsize=(14,15))
    gs = gridspec.GridSpec(2,2, width_ratios=(1,1), height_ratios=(4,1))

    targetColors = ['green', 'blue', 'cyan', 'darkorange', 'purple']

    col = 0
    row = 0

    for (path, prediction) in zip(paths, predictions):

        row %= 2 
        print('row, col:', row, col)

        ax = plt.subplot(gs[row, col])
        ax.set_xlim(scene.dimensions['xmin'], scene.dimensions['xmax'])
        ax.set_ylim(scene.dimensions['ymin'], scene.dimensions['ymax'])
        ax.set_aspect('equal')

        ax.set_xlabel('X-Position [m]')
        ax.set_ylabel('Y-Position [m]')

        for obstacle in scene.obstacles:

            obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
            ax.add_patch(obs)
     
        for i, target in enumerate(scene.targets):

            tar = plt.Circle((target[0], target[1]), target[2], color=targetColors[i], fill=False)
            ax.add_patch(tar)        
            ax.text(target[0]-0.5, target[1]+0.5, i)

        x, y = get_path(path)
        ax.scatter(x, y, color='blue')

        timeSteps = range(0, prediction.shape[0])

        row += 1
        pred_ax = plt.subplot(gs[row, col])
        pred_ax.set_ylim(0.0, 1.0)
        pred_ax.set_xlabel('Time step (1/100 s)')
        pred_ax.set_ylabel('Confidence')

        if col == 0:
            ax.set_title('Non-perturbed Optimal RRT Path')
            pred_ax.set_title('LSTM Confidence Given Non-perturbed Path')
        else:
            ax.set_title('Optimal RRT Path with Perturbation')
            pred_ax.set_title('LSTM Confidence Given Path with Perturbation')
    

        for targetIdx in range(len(targetColors)):

            conf = []

            for t in range(len(timeSteps)):

                conf.append(prediction[t, targetIdx])

            pred_ax.plot(timeSteps, conf, linestyle='--', color=targetColors[targetIdx])

        col += 1
        row += 1

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

def get_perturbation_target(labelIdx, noisyPredictions, timeStep=20):

    secondBest = 0.0 
    secondBestIdx = None

    for i in range(len(noisyPredictions[timeStep])):

        candidate = noisyPredictions[timeStep][i]
        print(i, secondBest, candidate)
        if i != labelIdx and secondBest < candidate:
            secondBestIdx = i
            secondBest = candidate

    return secondBestIdx 

def replan(modelSelection, dataDirectory, algorithm, attack='noise'):

    analyzer = get_gradient_analyzer(modelSelection, dataDirectory, algorithm)
    dataset = get_dataset(modelSelection, dataDirectory, algorithm)
    planner = AdversarialPlanner()
    scene = Scene('test_room')

    for instance, label in dataset.data:

        fooled = False
        gtTarget = np.argmax(label)

        origInstance = np.copy(instance)
        origGrads, origPredictions = analyzer.analyze(origInstance, label)
        attackedPath = instance 
        attackedPredictions = origPredictions
        grads = origGrads

        it = 0

        while not fooled:

            perturbTarget = get_perturbation_target(gtTarget, attackedPredictions)


            if attack == 'noise':
                attackedPath = planner.add_noise(grads, attackedPath, gtIdx=gtTarget, perturbIdx=perturbTarget)
            else:
                attackedPath = planner.one_pixel_attack(grads, attackedPath, gtIdx=gtTarget, perturbIdx=perturbTarget)

            grads, attackedPredictions = analyzer.analyze(attackedPath, label)


            print('perturb idx:', perturbTarget)
            print('gt idx:', gtTarget, flush=True)
            
            paths = [origInstance, attackedPath]
            preds = [origPredictions, attackedPredictions]

            plot_paths_and_predictions(paths, preds, scene, it)

            fooled = is_fooled(attackedPredictions, label)

            it += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--directory', type = str, default = '.\\data\\')
    parser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "RRT")
    parser.add_argument('--attack', type=str, help='Adversarial attack [noist/one_pixel.', default = "noise")

    args = parser.parse_args()

    modelSelection = args.model
    algorithm = args.algo.lower()
    dataDirectory = args.directory
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\'
    attack = args.attack

    replan(modelSelection, dataDirectory, algorithm, attack)
