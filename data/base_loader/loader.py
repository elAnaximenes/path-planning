import numpy as np
import json
import os
from tensorflow.keras.utils import to_categorical

class DataLoader:

    def __init__(self, split, numBatches, truncatedPathLength=1000):

        self.split = split
        self.numBatchesToLoad = int(numBatches)
        self.truncatedPathLength = truncatedPathLength 
        self.batches = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.trainData = None
        self.valData = None

    def get_mean_and_std(self, data):

        data = np.array(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return mean, std

    def _normalize_instances(self):

        mean = self.x_train.mean(axis=0, dtype=np.float64)
        self.x_train -= mean
        std = self.x_train.std(axis=0, dtype=np.float64)

        samples, features, timesteps = self.x_train.shape
        print(samples, features, timesteps)
        for i in range(samples):
            for j in range(features):
                for k in range(timesteps):
                    if std[j,k] > 0:
                        self.x_train[i,j,k] /= std[j,k]
        #self.x_train /= std

        self.x_val -= mean
        samples, features, timesteps = self.x_val.shape
        print(samples, features, timesteps)
        for i in range(samples):
            for j in range(features):
                for k in range(timesteps):
                    if std[j,k] > 0:
                        self.x_val[i,j,k] /= std[j,k]
        #self.x_val /= std

    def _combine_batches(self):

        self.x = self.batches[0][0]
        self.y = self.batches[0][1]

        for x_batch, y_batch in self.batches[1:]:

            self.x = np.concatenate((self.x, x_batch), axis=0)
            self.y = np.concatenate((self.y, y_batch))

    def _split_data(self):

        splitIndex = int(self.x.shape[0] * self.split)
        self.x_train = self.x[:splitIndex]
        self.y_train = self.y[:splitIndex]
        self.x_val = self.x[splitIndex:]
        self.y_val = self.y[splitIndex:]

    def _pre_process_data(self):

        self._combine_batches()

        # transform labels to one hot
        self.x = np.array(self.x)
        self.y = to_categorical(np.array(self.y))

        # split into train and test sets
        self._split_data()

        # normalize to center mean at zero
        self._normalize_instances()

        self.trainData = (self.x_train, self.y_train)
        self.valData = (self.x_val, self.y_val)

    def _load_batch_json(self, batchFileName):

        # template for subclasses
        print('Load batch from json is not implemented in subclass')
        pass

    def load(self):

        self.batches = []
        self.batchFileNames = os.listdir('./data/batches-train')

        for batchFileName in self.batchFileNames:

            x_batch, y_batch = self._load_batch_json(batchFileName)
            self.batches.append((x_batch, y_batch))

            self.numBatchesToLoad -= 1
            if self.numBatchesToLoad == 0:
                break

        self._pre_process_data()
        
        return (self.x_train, self.y_train), (self.x_val, self.y_val)

        


