import numpy as np
import json
import os
import tensorflow as tf
import math
from tensorflow.keras.utils import to_categorical
from .base_loaders.loaders import DataLoader 

class meanPathDataset:

    def __init__(self, stepSize = 1):

        self.pathsByLabel = {0:[], 1:[], 2:[], 3:[], 4:[]}
        self.meanPaths = {0:None, 1:None, 2:None, 3:None, 4:None}
        self.data = None
        self.stepSize = stepSize 

    def format_datasets_by_target(self):

        datasetsByTarget = {0:None, 1:None, 2:None, 3:None, 4:None}

        for target in datasetsByTarget:

            label = np.zeros(len(datasetsByTarget))
            label[target] = 1.0

            subDataset = meanPathDataset()
            data = []

            for path in self.pathsByLabel[target]:

                instance = np.transpose(path)
                instance = instance[np.newaxis, :, :] 
                pair = (instance, label)
                data.append(pair)

            subDataset.data = data
            datasetsByTarget[target] = subDataset

        return datasetsByTarget

class MeanPathDataLoader:

    def __init__(self, numBatches, dataDirectory, sampleNodes=100):

        self.numBatchesToLoad = int(numBatches)
        self.dataDirectory = dataDirectory
        self.sampleNodes = 100
        self.batches = None
        self.numSamples = 0
        self.x = None
        self.y = None
        self.dataset = meanPathDataset()

    def _package_data_for_testing(self):
        
        data = []

        for category in self.dataset.meanPaths:

            # one hot
            label = np.zeros(len(self.dataset.meanPaths))
            label[category] = 1.0

            # batch, time, features
            instance = np.transpose(self.dataset.meanPaths[category])
            instance = instance[np.newaxis, :, :] 
            instance[:, :, :2] /= 10.0
            instance[:, :, 2] -= math.pi
            instance[:, :, 2] /= math.pi

            # zip
            pair = (instance, label)
            data.append(pair)


        self.dataset.data = data

    def _calculate_mean_paths(self):

        for y in self.dataset.pathsByLabel:

            self.dataset.meanPaths[y] = np.mean(self.dataset.pathsByLabel[y], axis=0)
 
    def _sort_paths_into_bins(self):

        for x, y in zip(self.x, self.y):

            self.dataset.pathsByLabel[y].append(x)

        for y in self.dataset.pathsByLabel:

            self.dataset.pathsByLabel[y] = np.array(self.dataset.pathsByLabel[y])

    def _make_all_paths_same_length(self):

        x = []

        for path in self.x:

            stride = path.shape[1]//self.sampleNodes
            x.append(path[:, :(self.sampleNodes*stride):stride])

        self.x = x

    def _combine_batches(self):

        self.x = []
        self.y = [] 

        for x_batch, y_batch in self.batches:

            self.x += x_batch
            self.y += y_batch

    def _pre_process_data(self):

        self._combine_batches()

        self._make_all_paths_same_length()

        self._sort_paths_into_bins()

        self._calculate_mean_paths()

    def _load_batch_json(self, batchFileName):

        rawData = {}
        with open(os.path.join(self.dataDirectory, batchFileName), 'r') as f:
            rawData = json.load(f)

        instances = []
        labels = []

        for sampleNumber in range(len(rawData)):

            sample = rawData[str(sampleNumber)]

            x = sample['path']['x']
            y = sample['path']['y']
            theta = sample['path']['theta']

            instance = np.array([x, y, theta])
            label = sample['target']['index']

            instances.append(instance)
            labels.append(label)

        x_batch = instances
        y_batch = labels

        return (x_batch, y_batch)

    def load(self, startBatch=0):

        self.batches = []

        print('Loading data...')
        for i in range(startBatch, startBatch + self.numBatchesToLoad):

            batchFileName = 'test_room_batch_{}.json'.format(i)
            print(batchFileName)

            x_batch, y_batch = self._load_batch_json(batchFileName)
            self.batches.append((x_batch, y_batch))

            self.numBatchesToLoad -= 1
            if self.numBatchesToLoad == 0:
                break

        self._pre_process_data()

        self._package_data_for_testing()

        return self.dataset
