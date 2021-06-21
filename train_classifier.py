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

def build_model(modelSelection, inputShape):

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
 
def histogram_path_lengths(pathLengths):
     
    plt.hist(pathLengths, bins=50) 

    mean, std = get_mean_and_std(pathLengths)

    plt.axvline(mean, ls="--", color='red')
    plt.axvline(mean+std, ls="--", color='yellow')
    plt.axvline(mean-std, ls="--", color='yellow')
    
    plt.xlabel('Path Lengths')
    plt.ylabel('Number of Paths')
    plt.title('Distribution of path lengths')

    plt.show()

    return mean, std

def summary(history):

    plot_performance(history)
    """
    mean, std = histogram_path_lengths(pathLengths)

    print("minimum path length:", np.min(pathLengths))
    print("maximum path length:", np.max(pathLengths))
    print("mean path length:", mean)
    print("std path length:", std) 
    """

def normalize_instances(x_train, x_val):

    mean = x_train.mean(axis=0, dtype=np.float64)
    x_train -= mean
    std = x_train.std(axis=0, dtype=np.float64)

    x_train /= std

    x_val -= mean
    x_val /= std

    return x_train, x_val

def combine_batches(batches):

    x = batches[0][0]
    y = batches[0][1]

    for x_batch, y_batch in batches[1:]:

        x = np.concatenate((x, x_batch), axis=0)
        y = np.concatenate((y, y_batch))

    return x, y

def split_data(x, y, trainValSplit):

    splitIndex = int(x.shape[0] * trainValSplit)
    x_train = x[:splitIndex]
    y_train = y[:splitIndex]
    x_val = x[splitIndex:]
    y_val = y[splitIndex:]

    return (x_train, y_train), (x_val, y_val)

def pre_process_data(batches, trainValSplit, modelSelection):

    x, y = combine_batches(batches)

    # transform labels to one hot
    x = np.array(x)
    y = to_categorical(np.array(y))

    # split into train and test sets
    (x_train, y_train), (x_val, y_val) = split_data(x, y, trainValSplit)

    # normalize to center mean at zero
    x_train, x_val = normalize_instances(x_train, x_val)

    trainData = (x_train, y_train)
    valData = (x_val, y_val)

    return trainData, valData

def get_mean_and_std(data):

    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std

def load_batch_json(batchFileName, truncatedPathLength):

    # load raw json dict
    rawData = {}
    with open('./batches-train/{}'.format(batchFileName), 'r') as f:
        rawData = json.load(f)

    # build a list of instances and labels
    instances = [] 
    labels = [] 

    for sampleNumber in range(len(rawData)):

        sample = rawData[str(sampleNumber)]
        
        x = sample['path']['x']
        y = sample['path']['y']
        theta = sample['path']['theta']

        if len(x) < truncatedPathLength:
            instance = np.zeros((3,truncatedPathLength))
            x = np.array(x)
            y = np.array(y)
            theta = np.array(theta)
            instance[0, :x.shape[0]] = x
            instance[0, :y.shape[0]] = y
            instance[0, :theta.shape[0]] = theta
        else:
            instance = np.array([x[:truncatedPathLength],y[:truncatedPathLength],theta[:truncatedPathLength]])

        label = sample['target']['index']

        instances.append(instance)
        labels.append(label)

    x_batch = np.array(instances)
    y_batch = np.array(labels)

    return (x_batch, y_batch)

def load_batch_csv(batchFileName, batchSize):

    pass

def train_DNN(modelSelection, epochs, batchSize, split, numBatches):

    batches = []
    batchFileNames = os.listdir('./batches-train')
    truncatedPathLength = 1000 

    print('started loading data')
    
    # load raw data
    for batchFileName in batchFileNames:

        x_batch, y_batch = load_batch_json(batchFileName, truncatedPathLength)
        batches.append((x_batch, y_batch))

        numBatches -= 1
        if numBatches == 0:
            break

    print('finished loading data')

    # transform labels to categorical
    (x_train, y_train), (x_val, y_val) = pre_process_data(batches, split, modelSelection)

    # build classifier
    inputShape = x_train[0].shape
    model = build_model(modelSelection, inputShape)
    trainer = get_trainer(modelSelection, model)

    # fit model to training data
    history = trainer.train((x_train, y_train), (x_val, y_val), epochs, batchSize)

    print('input shape:', inputShape)
    print('number of paths in training set:', len(x_train))
    summary(history)

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='Deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--model', type=str, help='Which model to use for classification.', default = "FeedForward")
    parser.add_argument('--epochs', type=int, help='number of epochs to train for', default = 30)
    parser.add_argument('--batchsize', type=float, help='Size of batch in each epoch.', default = 512)
    parser.add_argument('--split', type=float, help='Percentage of set to use for training.', default = 0.95)
    parser.add_argument('--batches', type=float, help='How many batches to train on.', default = 10)

    args = parser.parse_args()
    modelSelection=args.model
    epochs = args.epochs
    batchSize = args.batchsize
    split = args.split
    numBatches = args.batches

    if split > 1.0 or split < 0.0:
        print('split must be a real number between 0.0 and 1.0')
        exit(2)

    # train
    train_DNN(modelSelection, epochs, batchSize, split, numBatches)
