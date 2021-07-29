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

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from feed_forward import FeedForward, FeedForwardTrainer
from lstm import LSTM, LSTMTrainer
from cnn import CNN, CNNTrainer
from data.loaders.feed_forward_train_loader import FeedForwardTrainDataLoader
from data.loaders.lstm_train_loader import LstmTrainDataLoader
from data.loaders.cnn_train_loader import CNNTrainDataLoader
from training_history.plot_training_history import plot_training_history

def build_model(modelSelection, inputShape):

    print('input shape:')

    if modelSelection.lower() == 'feedforward':

        model = FeedForward(inputShape)

    elif modelSelection.lower() == 'lstm':

        model = LSTM()

    elif modelSelection.lower() == 'cnn':

        model = CNN()
        
    return model

def get_trainer(modelSelection, model, weightsDir):

    if modelSelection.lower() == 'feedforward':

        trainer = FeedForwardTrainer(model, weightsDir)

    elif modelSelection.lower() == 'lstm':

        trainer = LSTMTrainer(model, weightsDir)

    elif modelSelection.lower() == 'cnn':
        
        trainer = CNNTrainer(model, weightsDir)

    return trainer

def get_data_loader(modelSelection, numBatches, dataDirectory, split, truncation, stepSize):

    if modelSelection.lower() == 'feedforward':

        dataLoader = FeedForwardTrainDataLoader(split, numBatches, dataDirectory = dataDirectory, truncatedPathLength=truncation, stepSize=stepSize)

    elif modelSelection.lower() == 'lstm':

        dataLoader = LstmTrainDataLoader(split, numBatches, dataDirectory = dataDirectory)

    elif modelSelection.lower() == 'cnn':

        dataLoader = CNNTrainDataLoader(split, numBatches, dataDirectory = dataDirectory, truncatedPathLength=truncation, stepSize=stepSize)

    return dataLoader

def plot_performance(history, modelSelection, algo):

    print('plotting performance')
    loss = history['trainLoss']
    valLoss = history['valLoss']

    epochs = range(1, len(loss) + 1)
    
    plt.plot(epochs, valLoss, 'bo', label = 'Validation loss')
    plt.plot(epochs, loss, 'b', label = 'Training loss')
    plt.title('Validation and training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('loss')
    plt.show()

    plt.clf()

    acc = history['trainAcc']
    valAcc = history['valAcc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, valAcc, 'bo', label = 'Validation acc')
    plt.plot(epochs, acc, 'b', label = 'Training acc')
    plt.title('{} Validation and training accuracy -- {}'.format(modelSelection, algo))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.savefig('loss')
    plt.show()

def summary(history, modelSelection, numBatches, startBatch, algo):

    plot_performance(history, modelSelection, algo)
    #plot_training_history(history, modelSelection)
    historyFileName = './training_history/{}_{}_batches_starting_at_{}.json'.format(modelSelection, numBatches, startBatch)
    trainingHistory = {'trainLoss': history['trainLoss'],\
                     'valLoss': history['valLoss'],\
                     'trainAcc': [acc.numpy().tolist() for acc in history['trainAcc']],\
                     'valAcc': [acc.numpy().tolist() for acc in history['valAcc']]}

    with open(historyFileName, 'w') as jsonFile:
        json.dump(trainingHistory, jsonFile)

def train_model(modelSelection, epochs, batchSize, split, numBatches, resume=False, startBatch=0, dataDirectory='./data/rrt-batches-train/', algo='rrt', sceneName='tower_defense', truncation=1, stepSize=100, predictorPlanner = 'planner'):

    # load training and validation data
    trainingDataDir = os.path.join(dataDirectory, '{}_dataset/{}_batches_train_{}'.format(sceneName, algo, predictorPlanner)) 
    dataLoader = get_data_loader(modelSelection, numBatches, trainingDataDir, split, truncation, stepSize)
    (x_train, y_train), (x_val, y_val) = dataLoader.load(startBatch) 
    print('number of paths in training set:', len(x_train))

    # build classifier
    inputShape = x_train[0].shape[1]
    model = build_model(modelSelection, inputShape)
    weightsDir = os.path.join(dataDirectory, '{}_dataset/{}_{}_weights_{}'.format(sceneName, algo, modelSelection.lower(), predictorPlanner))
    trainer = get_trainer(modelSelection, model, weightsDir)
    print('successfully loaded data and built model')

    # fit model to training data
    originalOut = sys.stdout
    history = trainer.train((x_train, y_train), (x_val, y_val), epochs, batchSize, resume)
    print('successfully trained model')

    # summarize training results
    # summary(history, modelSelection, numBatches, startBatch, algo)

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='Deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--epochs', type=int, help='number of epochs to train for', default = 30)
    parser.add_argument('--batch_size', type=int, help='Size of batch in each epoch.', default = 512)
    parser.add_argument('--split', type=float, help='Percentage of set to use for training.', default = 0.95)
    parser.add_argument('--batches', type=int, help='How many batches to train on.', default = 10)
    parser.add_argument('--resume',  dest='resume', action = 'store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('--start_batch', type = int, help='What batch number to start at', default = 0)
    parser.add_argument('--directory', type = str, default = '..\\data\\')
    parser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "RRT")
    parser.add_argument('--scene', type=str, help='Which scene to train over.', default = "tower_defense")
    parser.add_argument('--truncation', type=int, help='how much of the path to make visible to classifier', default=1)
    parser.add_argument('--step_size', type=int, help='Sample path every ? timesteps', default=100)
    parser.add_argument('--predictor', action='store_true', help='Is this LSTM is for planning or predicting', default=False)

    args = parser.parse_args()
    modelSelection=args.model
    epochs = args.epochs
    batchSize = args.batch_size
    split = args.split
    numBatches = args.batches
    resume = args.resume
    startBatch = args.start_batch
    dataDirectory = args.directory
    sceneName = args.scene
    truncation = args.truncation
    stepSize = args.step_size
    predictor = args.predictor
    predictorPlanner = 'planner'
    if predictor:
        predictorPlanner = 'predictor'

    if split > 1.0 or split < 0.0:
        print('split must be a real number between 0.0 and 1.0')
        exit(2)
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\'

    algorithm = args.algo.lower()

    # train
    train_model(modelSelection, epochs, batchSize, split, numBatches, resume, startBatch, dataDirectory, algorithm, sceneName, truncation, stepSize, predictorPlanner)
