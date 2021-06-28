import matplotlib.pyplot as plt
import os
import json
import numpy as np
import argparse

def plot_training_history(history, modelSelection):

    print('plotting performance')
    loss = history['trainLoss']
    valLoss = history['valLoss']

    epochs = range(1, len(loss) + 1)
    
    #plt.plot(epochs, valLoss, 'bo', label = 'Validation loss')
    plt.plot(epochs, loss, 'b', label = 'Training loss')
    plt.title('{} Training loss'.format(modelSelection))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss')
    plt.show()

    plt.clf()

    acc = history['trainAcc']
    valAcc = history['valAcc']

    epochs = range(1, len(acc) + 1)

    #plt.plot(epochs, valAcc, 'bo', label = 'Validation acc')
    plt.plot(epochs, acc, 'b', label = 'Training acc')
    plt.title('{} Training accuracy'.format(modelSelection))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy')
    plt.show()


def load_all_history(modelSelection, batchesPerHistory, startBatch, numHistoriesToLoad):

    allHistory = {'trainLoss':[], 'valLoss':[], 'trainAcc':[], 'valAcc':[]}

    for i in range(numHistoriesToLoad):

        newHistory = {}
        start = startBatch + (batchesPerHistory * i)
        historyFileName = './{}_{}_batches_starting_at_{}.json'.format(modelSelection, batchesPerHistory, start)

        with open(historyFileName) as f:
            newHistory = json.load(f)

        if i == 0:
            allHistory['trainLoss'] = np.array(newHistory['trainLoss'])
            allHistory['valLoss'] = np.array(newHistory['valLoss'])
            allHistory['trainAcc'] = np.array(newHistory['trainAcc'])
            allHistory['valAcc'] = np.array(newHistory['valAcc'])
        else:
            allHistory['trainLoss'] += np.array(newHistory['trainLoss'])
            allHistory['valLoss'] += np.array(newHistory['valLoss'])
            allHistory['trainAcc'] += np.array(newHistory['trainAcc'])
            allHistory['valAcc'] += np.array(newHistory['valAcc'])

    for key in allHistory:

        allHistory[key] /= numHistoriesToLoad

    return allHistory

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Retrieves training history over separate training batches, aggregates results and displays on one graph')
    parser.add_argument('--model', type=str, default='FeedForward')
    parser.add_argument('--histories', type=int, help='number of histories to aggregate (separate training jobs)', default = 5)
    parser.add_argument('--startbatch', type=int, default=0)
    parser.add_argument('--batches', type=int, help='Number of batches in each training history', default=40)

    args = parser.parse_args()
    modelSelection = args.model
    batchesPerHistory = args.batches 
    startBatch = args.startbatch 
    numHistoriesToLoad = args.histories 

    history = load_all_history(modelSelection, batchesPerHistory, startBatch, numHistoriesToLoad)
    plot_training_history(history, modelSelection)

