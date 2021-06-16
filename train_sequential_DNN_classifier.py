import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_performance(history):

    loss = history.history['loss']
    valLoss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)
    
    plt.plot(epochs, valLoss, 'bo', label = 'Validation loss')
    plt.plot(epochs, loss, 'b', label = 'Training loss')
    plt.title('Validation and training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()

    acc = history.history['accuracy']
    valAcc = history.history['val_accuracy']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, valAcc, 'bo', label = 'Validation acc')
    plt.plot(epochs, acc, 'b', label = 'Training acc')
    plt.title('Validation and training accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def build_model(inputShape):

    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(inputShape)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def normalize_instances(x_train, x_val):

    mean = x_train.mean(axis=0, dtype=np.float64)
    x_train -= mean
    std = x_train.std(axis=0, dtype=np.float64)

    x_train /= std

    x_val -= mean
    x_val /= std

    return x_train, x_val

def combine_training_batches(trainingBatches):

    x_train = trainingBatches[0][0]
    y_train = trainingBatches[0][1]
    for x_batch, y_batch in trainingBatches[1:]:

        x_train = np.concatenate((x_train, x_batch), axis=0)
        y_train = np.concatenate((y_train, y_batch))

    return x_train, y_train

def histogram_path_lengths(pathLengths):

    plt.hist(pathLengths, bins=50) 
    plt.xlabel('Path Lengths')
    plt.ylabel('Number of Paths')
    plt.title('Distribution of path lengths')

    plt.show()

def get_length_of_shortest_and_longest_paths(data):

    shortestPathLength = None
    longestPathLength = None
    for sampleNumber in range(len(data)):
        
        x = data[str(sampleNumber)]['path']['x']
        if shortestPathLength == None or shortestPathLength > len(x):
            shortestPathLength = len(x)
        if longestPathLength == None or longestPathLength < len(x):
            longestPathLength = len(x)

    print('shortest path: {}, longest Path: {}'.format(shortestPathLength, longestPathLength))

    return shortestPathLength, longestPathLength

def load_training_batch(sceneName, batchNumber, batchSize, truncatedPathLength):

    # load raw json dict
    rawData = {}
    batchFileName = './batches/{}_batch_{}.json'.format(sceneName, batchNumber)
    
    with open(batchFileName, 'r') as f:
        rawData = json.load(f)

    get_length_of_shortest_and_longest_paths(rawData)

    instances = []
    labels = [] 

    minLength = None

    batchPathLengths = []

    for sampleNumber in range(batchSize):

        sample = rawData[str(sampleNumber)]

        samplePathLength = len(sample['path']['x'])
        batchPathLengths.append(round(float(samplePathLength)/10.0) * 10)

        instance = np.zeros((3,truncatedPathLength), dtype = np.float64)
        for i in range(samplePathLength):

            if i == truncatedPathLength:
                break

            instance[0,i] = sample['path']['x'][i] 
            instance[1,i] = sample['path']['y'][i] 
            instance[2,i] = sample['path']['theta'][i] 

        #instances.append([sample['path']['x'][:truncatedPathLength], sample['path']['y'][:truncatedPathLength], sample['path']['theta'][:truncatedPathLength]])
        instances.append(instance)
        labels.append(sample['target']['index'])
        
    x_batch = np.array(instances)
    y_batch = to_categorical(np.array(labels))

    return (x_batch, y_batch), batchPathLengths

def train_DNN(sceneName, numBatches, batchSize):

    # load training data
    truncatedPathLength = 256 
    batches = []
    pathLengths = []

    for batchNumber in range(numBatches):

        data, batchPathLengths = load_training_batch(sceneName, batchNumber, batchSize, truncatedPathLength)
        pathLengths += batchPathLengths
        x_batch, y_batch = data
        batches.append((x_batch, y_batch))

    #histogram_path_lengths(pathLengths)

    # split batches into training and validate sets
    trainingBatches = batches[:-1]
    x_val, y_val = batches[-1]

    x_train, y_train = combine_training_batches(trainingBatches)

    x_train, x_val = normalize_instances(x_train, x_val)

    inputShape = x_train[0].shape
    model = build_model(inputShape)
    
    history = model.fit(x_train, y_train, epochs=30, batch_size=512, validation_data=(x_val, y_val))

    plot_performance(history)

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='sequential deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--scene', type=str, help='specify scene', default='simple_room')
    parser.add_argument('--batches', type=int, help='specify number of batches in dataset(one batch will be used for validation)', default=10)
    parser.add_argument('--batchsize', type=int, help='specify number of samples in each batch', default=100)
    args = parser.parse_args()

    # unpack args
    sceneName = args.scene
    numBatches = args.batches
    batchSize = args.batchsize

    if numBatches < 2:
        print("must specify more than one batch(one batch will be used for validation)")
        exit(2)

    # train
    train_DNN(sceneName, numBatches, batchSize)
