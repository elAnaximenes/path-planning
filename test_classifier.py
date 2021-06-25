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
from data.val_loader import ValidateDataLoader
from classifiers.lstm import LSTM, LSTMTester
from classifiers.feed_forward import FeedForward, FeedForwardTester

def plot_performance(performance):

    tp = performance['tp']
    labelCount = performance['label count']

    accuracyOverTime = []
    timesteps = []

    for i in range(len(labelCount)):
        percent = float(tp[i])/float(labelCount[i])
        accuracyOverTime.append(percent)
        timesteps.append(i)

    plt.title('Accuracy over time')
    plt.plot(timesteps, accuracyOverTime, 'b', label = 'Accuracy')
    plt.xlabel('timestep')
    plt.ylabel('% correct')
    plt.legend()
    plt.show()

def build_model(modelSelection, inputShape):

    print('input shape:', inputShape)

    if modelSelection.lower() == 'feedforward':

        model = FeedForward(inputShape)

    elif modelSelection.lower() == 'lstm':

        model = LSTM(inputShape)
        
    return model

def get_tester(modelSelection, dataset, model):

    if modelSelection.lower() == 'feedforward':

        tester = FeedForwardTester(dataset, model)

    elif modelSelection.lower() == 'lstm':

        tester = LSTMTester(dataset, model)

    return tester

def test_model(modelSelection, dataDirectory, numBatches):

    loader = None

    loader =ValidateDataLoader(numBatches, dataDirectory)
    dataset = loader.load()
    print('dataset loaded')

    inputShape = (3,)
    model = build_model(modelSelection, inputShape)
    print('model loaded')

    tester = get_tester(modelSelection, dataset, model) 
    print('tester loaded')

    performance = tester.test()

    plot_performance(performance)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Test selected model on validation or test set')
    argparser.add_argument('--model', type=str, default='FeedForward')
    argparser.add_argument('--directory', type=str, default='./data/batches-validate')
    argparser.add_argument('--batches', type=int, default=1)

    args = argparser.parse_args()

    modelSelection = args.model
    dataDirectory = args.directory

    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\batches-validate'
    numBatches = args.batches

    test_model(modelSelection, dataDirectory, numBatches)
