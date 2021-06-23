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
from classifiers.feed_forward import FeedForward, FeedForwardTrainer
from classifiers.lstm import LSTM, LSTMTrainer
from data.feed_forward_loader import FeedForwardDataLoader
from data.lstm_loader import LstmDataLoader

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

def get_data_loader(modelSelection):

    if modelSelection == 'FeedForward':
        dataLoader = FeedForwardDataLoader(split, numBatches)
    elif modelSelection == 'LSTM':
        dataLoader = LstmDataLoader(split, numBatches)

    return dataLoader

def plot_performance(history):

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

def summary(history):

    plot_performance(history)

def train_DNN(modelSelection, epochs, batchSize, split, numBatches, resume, startBatch):

    # load training and validation data
    dataLoader = get_data_loader(modelSelection)
    (x_train, y_train), (x_val, y_val) = dataLoader.load(startBatch) 
    print('number of paths in training set:', len(x_train))

    # build classifier
    #inputShape = x_train[0].shape[1]
    inputShape = (3,)
    model = build_model(modelSelection, inputShape)
    trainer = get_trainer(modelSelection, model)

    # fit model to training data
    history = trainer.train((x_train, y_train), (x_val, y_val), epochs, batchSize, resume)

    # summarize training results
    summary(history)

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='Deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--epochs', type=int, help='number of epochs to train for', default = 30)
    parser.add_argument('--batchsize', type=float, help='Size of batch in each epoch.', default = 512)
    parser.add_argument('--split', type=float, help='Percentage of set to use for training.', default = 0.95)
    parser.add_argument('--batches', type=float, help='How many batches to train on.', default = 10)
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
