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

    def __init__(self, instance, label, attackType, pointOfInterest = -1, perturbationRate=0.1):

        self.instance = instance
        self.label = label
        self.attackType = attackType
        self.pointOfInterest = pointOfInterest
        self.perturbationRate = perturbationRate

    def _get_most_salient_gradient(self, grads):

        gtIdx = np.argmax(self.label)

        # column of gt label grads and subtract this column from all other columns
        gtGrads = grads[:, gtIdx, :2].reshape(grads.shape[0], 1, 2)
        wrongGrads = np.delete(grads[:, :, :2], gtIdx, axis=1)
        summedGrads = wrongGrads - gtGrads

        # compute largest summed gradient
        summedGradsMagnitudes = np.linalg.norm(summedGrads, axis = 2)

        # get index of most salient gradient 
        mostSalientGradientIdx = np.unravel_index(np.argmax(summedGradsMagnitudes), summedGradsMagnitudes.shape)
        tMax, cMax = mostSalientGradientIdx

        # get x and y components of most salient gradient
        mostSalientGradient = summedGrads[tmax, cmax] 

        return mostSalientGradient, mostSalientGradientIdx

    def _one_pixel_attack(self, grads):

        mostSalientGradient, mostSalientGradientIdx = self._get_most_salient_gradient(grads)

        attackedPath = instance
        r = np.linalg.norm(instance[0, :, :] - instance[0, tMax], axis = 1)
        print(r.shape)
        exit(1)

        return attackedPath

    def _add_noise(self, grads, timeStepsToAttack):

        noisyPath = self.instance
        targetIdx = np.argmax(self.label)
        noisyPath[0, :timeStepsToAttack, :2] += (grads[:,targetIdx, :2] * self.perturbationRate)

        return noisyPath

    def perturb(self, grads):

        if self.attackType == 'noise':
            attackedPath = self._add_noise(grads, timeStepsToAttack=self.pointOfInterest)
        elif attack == 'one_pixel':
            attackedPath = self._one_pixel_attack(grads)

        return attackedPath

def plot_prediction(pred_ax, prediction):

    timeSteps = range(0, prediction.shape[0])
    pred_ax.set_ylim(0.0, 1.0)
    pred_ax.set_xlabel('Time step (1/100 s)')
    pred_ax.set_ylabel('Confidence')
    pred_ax.grid(True)

    targetColors = ['green', 'blue', 'cyan', 'darkorange', 'purple']
    for targetIdx in range(len(targetColors)):

        conf = []

        for t in range(len(timeSteps)):

            conf.append(prediction[t, targetIdx])

        pred_ax.plot(timeSteps, conf, linestyle='--', color=targetColors[targetIdx])

    return pred_ax

def get_path(path):

    x = []
    y = []

    for t in range(path.shape[1]):

        x.append(path[0,t, 0]*10)
        y.append(path[0,t, 1]*10)

    return x, y

def plot_path(ax, path, scene):

    ax.set_xlim(scene.dimensions['xmin'], scene.dimensions['xmax'])
    ax.set_ylim(scene.dimensions['ymin'], scene.dimensions['ymax'])
    ax.set_aspect('equal')

    ax.set_xlabel('X-Position [m]')
    ax.set_ylabel('Y-Position [m]')

    for obstacle in scene.obstacles:

        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        ax.add_patch(obs)
 
    targetColors = ['green', 'blue', 'cyan', 'darkorange', 'purple']
    for i, target in enumerate(scene.targets):

        tar = plt.Circle((target[0], target[1]), target[2], color=targetColors[i], fill=False)
        ax.add_patch(tar)        
        ax.text(target[0]-0.5, target[1]+0.5, i)

    x, y = get_path(path)
    ax.scatter(x, y, color='blue')

    return ax

def plot_paths_and_predictions(paths, predictions, scene, timeStep, it):
    
    fig = plt.figure(figsize=(14,15))
    gs = gridspec.GridSpec(2,2, width_ratios=(1,1), height_ratios=(4,1))

    col = 0
    row = 0

    for (path, prediction) in zip(paths, predictions):

        row %= 2 

        ax = plt.subplot(gs[row, col])
        ax = plot_path(ax, path, scene)

        row += 1
        pred_ax = plt.subplot(gs[row, col])
        pred_ax = plot_prediction(pred_ax, prediction)
        pred_ax.axvline(timeStep)

        if col == 0:
            ax.set_title('Non-perturbed Optimal RRT Path')
            pred_ax.set_title('LSTM Confidence Given Non-perturbed Path')
        else:
            ax.set_title('Optimal RRT Path with Perturbation')
            pred_ax.set_title('LSTM Confidence Given Path with Perturbation')

        col += 1
        row += 1

    plt.savefig('./data/saved_images/img-{}'.format(it))
    plt.show()

def is_fooled(predictions, label, timeStep):

    gtClass = np.argmax(label)
    predictedClass = np.argmax(predictions[timeStep])
    print('gt class is {}, prediction is {}'.format(gtClass, predictedClass))

    return gtClass != predictedClass 

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

def get_perturbation_target(labelIdx, noisyPredictions, timeStep):

    secondBest = 0.0 
    secondBestIdx = None

    for i in range(len(noisyPredictions[timeStep])):

        candidate = noisyPredictions[timeStep][i]
        if i != labelIdx and secondBest < candidate:
            secondBestIdx = i
            secondBest = candidate

    return secondBestIdx 

def replan(modelSelection, dataDirectory, algorithm, attack='noise'):

    analyzer = get_gradient_analyzer(modelSelection, dataDirectory, algorithm)
    dataset = get_dataset(modelSelection, dataDirectory, algorithm)

    scene = Scene('test_room')
    windowOfInterest = 0.99 

    for instance, label in dataset.data:

        timeStepsToAttack = int(instance.shape[1] * windowOfInterest) 

        planner = AdversarialPlanner(instance, label, attack, pointOfInterest=timeStepsToAttack)

        grads, origPredictions = analyzer.analyze(origInstance, label, timeStepsToAttack)
        origInstance = np.copy(instance)

        fooled = False
        it = 0
        while not fooled:

            attackedPath = planner.perturb(grads)
            grads, attackedPredictions = analyzer.analyze(attackedPath, label, timeStepsToAttack)
            fooled = is_fooled(attackedPredictions, label, timeStepsToAttack)

            if True:
                paths = [origInstance, attackedPath]
                preds = [origPredictions, attackedPredictions]
                plot_paths_and_predictions(paths, preds, scene, timeStepsToAttack, it)

            it += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--directory', type = str, default = '.\\data\\')
    parser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "RRT")
    parser.add_argument('--attack', type=str, help='Adversarial attack [noist/one_pixel.', default = "noise")

    args = parser.parse_args()

    files = os.listdir('./data/saved_images/')
    for f in files:
        os.remove(os.path.join('./data/saved_images/',f))

    modelSelection = args.model
    algorithm = args.algo.lower()
    dataDirectory = args.directory
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\'
    attack = args.attack

    replan(modelSelection, dataDirectory, algorithm, attack)
