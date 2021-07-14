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

        origInstance = np.copy(instance)
        grads, origPredictions = self.analyzer.analyze(origInstance, label, timeStepsToAttack)

        attackedPath = None
        fooled = False

        while not fooled:

            attackedPath = self.planner.perturb(grads)
            grads, attackedPredictions = self.analyzer.analyze(attackedPath, label, timeStepsToAttack)
            fooled = self._is_classifier_fooled(attackedPredictions, label, timeStepsToAttack)

            if True:
                paths = [origInstance, attackedPath]
                preds = [origPredictions, attackedPredictions]
                self.visualizer.plot_paths_and_predictions(paths, preds, timeStepsToAttack)
        
        return attackedPath

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
    windowOfInterest = 0.99 
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

    attack_paths_in_batch(modelSelection, dataDirectory, algorithm, attack)
