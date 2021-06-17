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
    plt.savefig('loss')
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

def summary(history, pathLengths):

    plot_performance(history)
    mean, std = histogram_path_lengths(pathLengths)

    print("minimum path length:", np.min(pathLengths))
    print("maximum path length:", np.max(pathLengths))
    print("mean path length:", mean)
    print("std path length:", std) 
    print(history.model.summary())
    print(history.params)

def build_model(inputShape):

    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(inputShape)))
    model.add(layers.Flatten())
    model.add(layers.Dense(8192, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

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

def pre_process_data(batches, trainValSplit, minPathLength, maxPathLength):

    x = [] 
    y = []

    for x_batch, y_batch in batches:

        validX = []
        validY = []
        for instance, label in zip(x_batch, y_batch): 
            
            # throw out paths that are too long or too short
            pathLength = len(instance[0])
            if pathLength < minPathLength or pathLength > maxPathLength:
                continue

            # truncate paths so that we have a fixed length input 
            instance[0] = instance[0][:minPathLength]
            instance[1] = instance[1][:minPathLength]
            instance[2] = instance[2][:minPathLength]

            # these are the paths we are going to use
            validX.append(instance)
            validY.append(label)

        x += validX
        y += validY 

    # transform labels to one hot
    x = np.array(x)
    y = to_categorical(np.array(y))

    # split into train and test sets
    (x_train, y_train), (x_val, y_val) = split_data(x, y, trainValSplit)

    # normalize to center mean at zero
    x_train, x_val = normalize_instances(x_train, x_val)

    return (x_train, y_train), (x_val, y_val)

def get_mean_and_std(data):

    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std

def load_training_batch(sceneName, batchNumber):

    # load raw json dict
    rawData = {}
    batchFileName = './batches/{}_batch_{}.json'.format(sceneName, batchNumber)
    with open(batchFileName, 'r') as f:
        rawData = json.load(f)

    # build a list of instances and labels
    instances = []
    labels = [] 
    batchPathLengths = []

    for sampleNumber in range(len(rawData)):

        sample = rawData[str(sampleNumber)]

        samplePathLength = len(sample['path']['x'])
        batchPathLengths.append(round(float(samplePathLength)/10.0) * 10)

        instance = [sample['path']['x'], sample['path']['y'], sample['path']['x']]
        instances.append(instance)
        labels.append(sample['target']['index'])
        
    x_batch = instances
    y_batch = labels

    return (x_batch, y_batch), batchPathLengths

def train_DNN(sceneName, numBatches):

    batches = []
    pathLengths = []

    # load raw data
    for batchNumber in range(numBatches):

        data, batchPathLengths = load_training_batch(sceneName, batchNumber)
        pathLengths += batchPathLengths
        x_batch, y_batch = data
        batches.append((x_batch, y_batch))

    # draw histogram of path length distributions
    mean, std = get_mean_and_std(pathLengths)
 
    # set params for longest and shortest path
    minPathLength = int(mean - std)
    maxPathLength = int(mean + (2*std))

    trainValSplit = 0.95
    # drop paths that are extremely long or short and transform labels to categorical
    (x_train, y_train), (x_val, y_val) = pre_process_data(batches, trainValSplit, minPathLength, maxPathLength)

    # build classifier
    inputShape = x_train[0].shape
    model = build_model(inputShape)
    
    history = model.fit(x_train, y_train, epochs=1, batch_size=512, validation_data=(x_val, y_val))

    summary(history, pathLengths)

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='sequential deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--scene', type=str, help='specify scene', default='simple_room')
    parser.add_argument('--batches', type=int, help='specify number of batches in dataset', default=10)
    args = parser.parse_args()

    # unpack args
    sceneName = args.scene
    numBatches = args.batches

    # train
    train_DNN(sceneName, numBatches)
