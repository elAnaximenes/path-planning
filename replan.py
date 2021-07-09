import tensorflow as tf
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from data.val_loader import ValidateDataLoader
from classifiers.lstm import LSTM, LSTMGradientAnalyzer
from dubins_path_planner.RRT import Scene
import visualize_gradients

class AdversarialPlanner:

    def __init__(self):
        
        pass

    def add_noise(self, grads, instance):

        #instance is (1, timesteps, features)

        pass

def is_fooled(predictions, label):

    print(predictions.shape)
    print(label.shape)
    exit(1)

    return True 

def get_dataset(modelSelection, dataDirectory, algorithm):

    dataset = None 
    
    valDataDir = os.path.join(dataDirectory, '{}_batches_validate'.format(algorithm)) 
    if modelSelection.lower() == 'lstm':
        loader = ValidateDataLoader(numBatches=1, dataDirectory=valDataDir)

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

    for instance, label in dataset:

        fooled = False
        grads, originalPredictions, instance = analyzer.analyze(origInstance, label)
        origInstance = np.copy(path)

        while not fooled:

            noisyPath, noisyPredictions = planner.add_noise(grads, instance)
            plot_paths_and_predictions((origPath, originalPredictions), (noisyPath, noisyPredictions))
            grads, predictions, instance = analyzer.analyze(instance, label, transform=False)
            fooled = is_fooled(predictions, label)

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
