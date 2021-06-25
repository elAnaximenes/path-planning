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
from data.feed_forward_train_loader import FeedForwardTrainDataLoader
from data.lstm_train_loader import LstmTrainDataLoader

def build_model(modelSelection, inputShape):

    print('input shape:', inputShape)

    if modelSelection == 'FeedForward':

        model = FeedForward(inputShape)

    elif modelSelection == 'LSTM':

        model = LSTM(inputShape)
        
    return model

def get_trainer(modelSelection, model):

    if modelSelection == 'FeedForward':

        trainer = FeedForwardTrainer(model)

    elif modelSelection == 'LSTM':

        trainer = LSTMTrainer(model)

    return trainer

def get_data_loader(modelSelection, numBatches):

    dataDirectory = './data/batches-train'
    if modelSelection == 'FeedForward':
        dataLoader = FeedForwardTrainDataLoader(split, numBatches, dataDirectory = dataDirectory)
    elif modelSelection == 'LSTM':
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
    plt.savefig('loss')
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
    plt.savefig('loss')
    plt.show()

def summary(history, modelSelection, numBatches, startBatch):

    plot_performance(history)
    historyFileName = './training_history/{}_{}_batches_starting_at_{}.json'.format(modelSelection, numBatches, startBatch)
    trainingHistory = {'trainLoss': history['trainLoss'],\
                     'valLoss': history['valLoss'],\
                     'trainAcc': [acc.numpy().tolist() for acc in history['trainAcc']],\
                     'valAcc': [acc.numpy().tolist() for acc in history['valAcc']]}

    with open(historyFileName, 'w') as jsonFile:
        json.dump(trainingHistory, jsonFile)

def train_DNN(modelSelection, epochs, batchSize, split, numBatches, resume, startBatch):

    # load training and validation data
    dataLoader = get_data_loader(modelSelection, numBatches)
    (x_train, y_train), (x_val, y_val) = dataLoader.load(startBatch) 
    print('number of paths in training set:', len(x_train))

    # build classifier
    #inputShape = x_train[0].shape[1]
    inputShape = (3,)
    model = build_model(modelSelection, inputShape)
    trainer = get_trainer(modelSelection, model)

    print('successfully loaded data and built model')
    # fit model to training data
    originalOut = sys.stdout
    sys.stdout = open('./logs/{}-training.log'.format(modelSelection), 'w')
    history = trainer.train((x_train, y_train), (x_val, y_val), epochs, batchSize, resume)
    sys.stdout = originalOut
    print('successfully trained model')

    # summarize training results
    summary(history, modelSelection, numBatches, startBatch)

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='Deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--epochs', type=int, help='number of epochs to train for', default = 30)
    parser.add_argument('--batchsize', type=float, help='Size of batch in each epoch.', default = 512)
    parser.add_argument('--split', type=float, help='Percentage of set to use for training.', default = 0.95)
    parser.add_argument('--batches', type=int, help='How many batches to train on.', default = 10)
    parser.add_argument('--resume',  dest='resume', action = 'store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('--startbatch', type = int, help='What batch number to start at', default = 0)

    args = parser.parse_args()
    modelSelection=args.model
    epochs = args.epochs
    batchSize = args.batchsize
    split = args.split
    numBatches = args.batches
    resume = args.resume
    startBatch = args.startbatch

    if split > 1.0 or split < 0.0:
        print('split must be a real number between 0.0 and 1.0')
        exit(2)

    # train
    train_DNN(modelSelection, epochs, batchSize, split, numBatches, resume, startBatch)
