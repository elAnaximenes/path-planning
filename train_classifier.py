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
from classifiers.feed_forward import FeedForward, FeedForwardTrainer
from classifiers.lstm import LSTM, LSTMTrainer
#from classifiers.cnn import CNN, CNNTrainer
from data.feed_forward_train_loader import FeedForwardTrainDataLoader
from data.lstm_train_loader import LstmTrainDataLoader
from training_history.plot_training_history import plot_training_history

def build_model(modelSelection):

    print('input shape:')

    if modelSelection.lower() == 'feedforward':

        model = FeedForward()

    elif modelSelection.lower() == 'lstm':

        model = LSTM()

    #elif modelSelection.lower() == 'cnn':

        #model = CNN()
        
    return model

def get_trainer(modelSelection, model, weightsDir):

    if modelSelection.lower() == 'feedforward':

        trainer = FeedForwardTrainer(model, weightsDir)

    elif modelSelection.lower() == 'lstm':

        trainer = LSTMTrainer(model, weightsDir)

    #elif modelSelection.lower() == 'cnn':
        
        #trainer = CNNTrainer(model, weightsDir)

    return trainer

def get_data_loader(modelSelection, numBatches, dataDirectory):

    if modelSelection.lower() == 'feedforward':

        dataLoader = FeedForwardTrainDataLoader(split, numBatches, dataDirectory = dataDirectory)

    elif modelSelection.lower() == 'lstm' or modelSelection.lower() == 'cnn':

        dataLoader = LstmTrainDataLoader(split, numBatches, dataDirectory = dataDirectory)

    return dataLoader

def plot_performance(history):

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
    plt.title('Validation and training accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.savefig('loss')
    plt.show()

def summary(history, modelSelection, numBatches, startBatch):

    #plot_performance(history)
    plot_training_history(history, modelSelection)
    historyFileName = './training_history/{}_{}_batches_starting_at_{}.json'.format(modelSelection, numBatches, startBatch)
    trainingHistory = {'trainLoss': history['trainLoss'],\
                     'valLoss': history['valLoss'],\
                     'trainAcc': [acc.numpy().tolist() for acc in history['trainAcc']],\
                     'valAcc': [acc.numpy().tolist() for acc in history['valAcc']]}

    with open(historyFileName, 'w') as jsonFile:
        json.dump(trainingHistory, jsonFile)

def train_model(modelSelection, epochs, batchSize, split, numBatches, resume=False, startBatch=0, dataDirectory='./data/rrt-batches-train/', algo='rrt'):

    # load training and validation data
    trainingDataDir = os.path.join(dataDirectory, '{}_batches_train'.format(algo)) 
    dataLoader = get_data_loader(modelSelection, numBatches, trainingDataDir)
    (x_train, y_train), (x_val, y_val) = dataLoader.load(startBatch) 
    print('number of paths in training set:', len(x_train))

    # build classifier
    #inputShape = x_train[0].shape[1]
    model = build_model(modelSelection)
    weightsDir = os.path.join(dataDirectory, '{}_{}_weights'.format(algo, modelSelection.lower()))
    trainer = get_trainer(modelSelection, model, weightsDir)

    print('successfully loaded data and built model')
    # fit model to training data
    originalOut = sys.stdout
    # sys.stdout = open('./logs/{}-training.log'.format(modelSelection), 'w')
    history = trainer.train((x_train, y_train), (x_val, y_val), epochs, batchSize, resume)
    # sys.stdout = originalOut
    print('successfully trained model')

    # summarize training results
    summary(history, modelSelection, numBatches, startBatch)

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='Deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--epochs', type=int, help='number of epochs to train for', default = 30)
    parser.add_argument('--batchsize', type=int, help='Size of batch in each epoch.', default = 512)
    parser.add_argument('--split', type=float, help='Percentage of set to use for training.', default = 0.95)
    parser.add_argument('--batches', type=int, help='How many batches to train on.', default = 10)
    parser.add_argument('--resume',  dest='resume', action = 'store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('--startbatch', type = int, help='What batch number to start at', default = 0)
    parser.add_argument('--directory', type = str, default = '.\\data\\')
    parser.add_argument('--algo', type=str, help='Which path planning algorithm dataset to train over.', default = "RRT")

    args = parser.parse_args()
    modelSelection=args.model
    epochs = args.epochs
    batchSize = args.batchsize
    split = args.split
    numBatches = args.batches
    resume = args.resume
    startBatch = args.startbatch
    dataDirectory = args.directory

    if split > 1.0 or split < 0.0:
        print('split must be a real number between 0.0 and 1.0')
        exit(2)
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\'

    algorithm = args.algo.lower()

    # train
    train_model(modelSelection, epochs, batchSize, split, numBatches, resume, startBatch, dataDirectory, algorithm)
