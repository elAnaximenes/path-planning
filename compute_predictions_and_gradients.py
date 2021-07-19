import json
import csv
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import json
import matplotlib.pyplot as plt
from dubins_path_planner.scene import Scene
from data.mean_path_loader import MeanPathDataLoader 
from classifiers.lstm import LSTM, LSTMGradientAnalyzer

def save_predictions(target, preds):
    
    fileName = './predictions/preds_{}.csv'.format(target)
    with open(fileName, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for pred in preds:
            writer.writerow(pred.tolist())

def save_gradients(target, grads):

    fileName = './gradients/grads_{}.csv'.format(target)
    with open(fileName, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for grad in grads:
            writer.writerow(grad.tolist())

def get_dataset(dataDir, algo, numBatches):

    valDataDir = os.path.join(dataDir, '{}_batches_train'.format(algo))
    loader = MeanPathDataLoader(numBatches, valDataDir)
    meanPathsData = loader.load()

    return meanPathsData

def get_gradient_analyzer(modelSelection, dataDirectory, algorithm):

    weightsDir = os.path.join(dataDirectory, '{}_{}_weights'.format(algorithm, modelSelection.lower()))
    analyzer = None

    if modelSelection.lower() == 'lstm':
        model = LSTM()
        analyzer = LSTMGradientAnalyzer(model, weightsDir)
        print('got analyzer')

    return analyzer

def get_model(modelSelection):

    if modelSelection == 'lstm':
        model = LSTM()

    return model

def visualize_confidence_surface(dataDir, algo, numBatches, modelSelection, target):

    sceneName = 'test_room'
    scene = Scene(sceneName)
    model = get_model(modelSelection)
    analyzer = get_gradient_analyzer(modelSelection, dataDir, algo)
    meanPathsData = get_dataset(dataDir, algo, numBatches)

    datasetsByTarget = meanPathsData.format_datasets_by_target()

    predictionsByTarget = {0:[], 1:[], 2:[], 3:[], 4:[]}
    gradsByTarget = {0:[], 1:[], 2:[], 3:[], 4:[]}

    i = 0
    for instance, label in datasetsByTarget[target].data:
            
        grads, preds = analyzer.analyze(instance, label, 99)
        gradsByTarget[target].append(grads)
        predictionsByTarget[target].append(preds)

    save_predictions(target, predictionsByTarget[target])
    save_gradients(target, gradsByTarget[target])
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--directory', type=str, default = './data/batches-train')
    parser.add_argument('--algo', type=str, help='Planning algorithm', default='rrt')
    parser.add_argument('--batches', type=int, help='number of training batches to load', default=10)
    parser.add_argument('--model', type=str, default = 'lstm')
    parser.add_argument('--target', type=int, default = 0)

    args = parser.parse_args()

    dataDir = args.directory
    if dataDir == 'tower':
        dataDir = 'D:\\path_planning_data\\'

    algo = args.algo
    numBatches = args.batches
    modelSelection = args.model.lower()
    target = args.target

    visualize_confidence_surface(dataDir, algo, numBatches, modelSelection, target)
