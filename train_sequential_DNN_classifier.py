import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import json
import numpy as np

def load_training_batch(sceneName, batchNumber, batchSize, truncatedPathLength):

    # load raw json dict
    rawData = {}
    batchFileName = './batches/{}_batch_{}.json'.format(sceneName, batchNumber)
    with open(batchFileName, 'r') as f:
        rawData = json.load(f)

    instances = []
    labels = [] 

    for sampleNumber in range(batchSize):

        sample = rawData[str(sampleNumber)]
             
        instances.append([sample['path']['x'][:truncatedPathLength], sample['path']['y'][:truncatedPathLength], sample['path']['theta'][:truncatedPathLength]])
        labels.append(sample['target']['index'])
        
    x_batch = np.array(instances)
    y_batch = np.array(labels)

    return x_batch, y_batch 

def build_model():

    model = keras.Sequential()

    return model

def train_DNN(sceneName, numBatches, batchSize):

    # load training data
    truncatedPathLength = 50000
    batches = []
    for batchNumber in range(numBatches):

        data = load_training_batch(sceneName, batchNumber, batchSize, truncatedPathLength)
        x_batch, y_batch = data
        batches.append((x_batch, y_batch))

    # split batches into training and validate sets
    trainingBatches = batches[:-1]
    validateBatch = batches[-1]
    
    model = build_model()

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='sequential deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--scene', type=str, help='specify scene', default='simple_room')
    parser.add_argument('--batches', type=int, help='specify number of batches to train over', default=1)
    parser.add_argument('--batchsize', type=int, help='specify number of samples in each batch', default=100)
    args = parser.parse_args()

    # unpack args
    sceneName = args.scene
    numBatches = args.batches
    batchSize = args.batchsize

    # train
    train_DNN(sceneName, numBatches, batchSize)





