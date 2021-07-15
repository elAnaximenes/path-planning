import tensorflow as tf
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib import gridspec
from data.val_loader import ValidateDataLoader
from classifiers.lstm import LSTM, LSTMGradientAnalyzer
from dubins_path_planner.scene import Scene
import visualize_gradients

class Adversary:

    def __init__(self, analyzer, scene, windowOfInterest):

        self.analyzer = analyzer
        self.scene = scene
        self.windowOfInterest = windowOfInterest
        self.planner = None
        self.visualizer = AttackVisualizer(self.scene)

    def _is_classifier_fooled(self, predictions, label, timeStep):

        gtClass = np.argmax(label)
        predictedClass = np.argmax(predictions[timeStep])
        print('gt class is {}, prediction is {}'.format(gtClass, predictedClass))

        return gtClass != predictedClass 

    def attack(self, instance, label, attackType='noise'):

        timeStepsToAttack = int(instance.shape[1] * self.windowOfInterest) 

        self.planner = AdversarialPlanner(instance, label, attackType, pointOfInterest=timeStepsToAttack)

        grads, origPredictions = self.analyzer.analyze(instance, label, timeStepsToAttack)
        origGrads = np.copy(grads)

        attackedPath = None
        fooled = False

        while not fooled:

            if attackType == 'one_pixel':
                grads = origGrads

            attackedPath = self.planner.perturb(grads)
            grads, attackedPredictions = self.analyzer.analyze(attackedPath, label, timeStepsToAttack)
            fooled = self._is_classifier_fooled(attackedPredictions, label, timeStepsToAttack)

            if True:
                paths = [self.planner.origInstance, attackedPath]
                preds = [origPredictions, attackedPredictions]
                self.visualizer.plot_paths_and_predictions(paths, preds, timeStepsToAttack, self.planner.it-2)
        
        return attackedPath

class AdversarialPlanner:

    def __init__(self, instance, label, attackType, pointOfInterest = -1, perturbationRate=0.1):

        self.instance = instance
        self.origInstance = np.copy(instance)
        self.label = label
        self.attackType = attackType
        self.pointOfInterest = pointOfInterest
        self.perturbationRate = perturbationRate
        self.it = 1

    def _get_most_salient_point(self, grads):

        gtIdx = np.argmax(self.label)

        # column of gt label grads and subtract this column from all other columns
        gtGrads = grads[:, gtIdx, :2]

        # get index of most salient gradient 
        gradMagnitudes = np.linalg.norm(gtGrads, axis=1)
        mostSalientTimeStep = np.argmax(gradMagnitudes[:self.pointOfInterest])

        # get x and y components of most salient gradient
        mostSalientGradient = gtGrads[mostSalientTimeStep] * self.perturbationRate

        return mostSalientGradient, mostSalientTimeStep

    def _one_pixel_attack(self, grads):

        mostSalientGradient, mostSalientTimeStep = self._get_most_salient_point(grads)
        gtIdx = np.argmax(self.label)

        # euclidean distance from point of attack
        r = np.linalg.norm(self.instance[0, :, :] -self.instance[0, mostSalientTimeStep, :], axis = 1)

        # dimension matching
        mostSalientGradient = np.tile(mostSalientGradient, (r.shape[0],1))
        r = np.transpose(np.tile(r, (2,1)))
        
        # attack and smooth
        attackedPath = np.copy(self.origInstance)
        attackedPath[0, :, :2] -= (np.exp(-5.0 * r) * mostSalientGradient * self.it )

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
        
        self.it += 1

        return attackedPath

class AttackVisualizer:

    def __init__(self, scene):

        self.scene = scene
        self.targetColors = ['green', 'blue', 'cyan', 'darkorange', 'purple']
        self.fig = None
        self.gs = None
        self.row = None
        self.col = None

    def _setup_prediction_ax(self, timeStepToAttack):
    
        pred_ax = plt.subplot(self.gs[self.row, self.col])
        pred_ax.set_ylim(0.0, 1.0)
        pred_ax.set_xlabel('Time step (1/100 s)')
        pred_ax.set_ylabel('Confidence')
        pred_ax.grid(True)
        pred_ax.axvline(timeStepToAttack)

        if self.col == 0:
            pred_ax.set_title('LSTM Confidence Given Non-perturbed Path')
        else:
            pred_ax.set_title('LSTM Confidence Given Path with Perturbation')

        return pred_ax

    def _plot_prediction(self, prediction, timeStepToAttack):

        pred_ax = self._setup_prediction_ax(timeStepToAttack)

        timeSteps = range(0, prediction.shape[0])

        for targetIdx in range(len(self.targetColors)):

            conf = []

            for t in range(len(timeSteps)):

                conf.append(prediction[t, targetIdx])

            pred_ax.plot(timeSteps, conf, linestyle='--', color=self.targetColors[targetIdx])

    def _get_path(self, path):

        x = []
        y = []

        for t in range(path.shape[1]):

            x.append(path[0,t, 0]*10)
            y.append(path[0,t, 1]*10)

        return x, y

    def _plot_targets(self, ax):

        for i, target in enumerate(self.scene.targets):

            tar = plt.Circle((target[0], target[1]), target[2], color=self.targetColors[i], fill=False)
            ax.add_patch(tar)        
            ax.text(target[0]-0.5, target[1]+0.5, i)
        
        return ax

    def _plot_obstacles(self, ax):

        for obstacle in self.scene.obstacles:

            obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
            ax.add_patch(obs)

        return ax

    def _plot_scene(self, ax):
    
        ax = self._plot_obstacles(ax)
        ax = self._plot_targets(ax)
             
        return ax

    def _setup_path_ax(self):

        ax = plt.subplot(self.gs[self.row, self.col])
        ax.set_xlim(self.scene.dimensions['xmin'], self.scene.dimensions['xmax'])
        ax.set_ylim(self.scene.dimensions['ymin'], self.scene.dimensions['ymax'])
        ax.set_aspect('equal')
        
        return ax

    def _plot_path(self, path):

        ax = self._setup_path_ax()

        if self.col == 0:
            ax.set_title('Non-perturbed Optimal RRT Path')
        else:
            ax.set_title('Optimal RRT Path with Perturbation')

        ax.set_xlabel('X-Position [m]')
        ax.set_ylabel('Y-Position [m]')

        ax = self._plot_scene(ax)

        x, y = self._get_path(path)
        ax.scatter(x, y, color='blue')

    def plot_paths_and_predictions(self, paths, predictions, timeStepToAttack, it):
        
        self.fig = plt.figure(figsize=(14,15))
        self.gs = gridspec.GridSpec(2,2, width_ratios=(1,1), height_ratios=(4,1))

        self.row = 0
        self.col = 0

        for (path, prediction) in zip(paths, predictions):

            self._plot_path(path)

            self.row += 1

            self._plot_prediction(prediction, timeStepToAttack)

            self.col += 1
            self.row = 0 

        plt.savefig('./data/saved_images/img-{}.png'.format(it))
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

def attack_paths_in_batch(modelSelection, dataDirectory, algorithm, attackType='noise'):

    analyzer = get_gradient_analyzer(modelSelection, dataDirectory, algorithm)
    scene = Scene('test_room')
    windowOfInterest = 0.90 
    adversary = Adversary(analyzer, scene, windowOfInterest)

    dataset = get_dataset(modelSelection, dataDirectory, algorithm)
    adversarialPaths = []

    for instance, label in dataset.data:

        attackedPath = adversary.attack(instance, label, attackType)
        adversarialPaths.append(attackedPath)

    return adversarialPaths

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--directory', type = str, default = '.\\data\\')
    parser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "RRT")
    parser.add_argument('--attack', type=str, help='Adversarial attack [noist/one_pixel].', default = "noise")

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

    attack_paths_in_batch(modelSelection, dataDirectory, algorithm, attack)
