import numpy as np
import json
import os
import math
from tensorflow.keras.utils import to_categorical

class DataLoader:

    def __init__(self, numBatches, truncatedPathLength=1000, dataDirectory='./data/batches-train/'):


        self.numBatchesToLoad = int(numBatches)
        print(self.numBatchesToLoad)
        self.truncatedPathLength = truncatedPathLength 
        self.dataDirectory = dataDirectory
        self.batches = None
        self.numSamples = 0
        self.x = None
        self.y = None

    def get_mean_and_std(self, data):

        data = np.array(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        print('data shape', data.shape)
        print('mean', mean)
        print('std', std)

        return mean, std

    def _normalize_instances(self):

        #mean = self.x_train.mean(axis=0, dtype=np.float64)
        #self.x_train -= mean
        #std = self.x_train.std(axis=0, dtype=np.float64)

        print('\nShape of data set:')
        print(self.x.shape)

        print('\nmaximum x value before normalization:', np.amax(self.x[:,0,:]))
        print('minimum x value before normalization:', np.amin(self.x[:,0,:]))
        print('maximum y value before normalization:', np.amax(self.x[:,1,:]))
        print('minimum y value before normalization:', np.amin(self.x[:,1,:]))
        self.x[:,:2,:] /= 10.0

        print('\nmaximum x value after normalization:', np.amax(self.x[:,0,:]))
        print('minimum x value after normalization:', np.amin(self.x[:,0,:]))
        print('maximum y value after normalization:', np.amax(self.x[:,1,:]))
        print('minimum y value after normalization:', np.amin(self.x[:,1,:]))

        print('\nmaximum theta value before normalization:', np.amax(self.x[:,2,:]))
        print('minimum theta value before normalization:', np.amin(self.x[:,2,:]))
        self.x[:,2,:] -= (math.pi)
        self.x[:,2,:] /= (math.pi)
        print('maximum theta value after normalization:', np.amax(self.x[:,2,:]))
        print('minimum theta value after normalization:', np.amin(self.x[:,2,:]))
        print('\n')

        """
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
        """

    def _combine_batches(self):

        self.x = self.batches[0][0]
        self.y = self.batches[0][1]

        for x_batch, y_batch in self.batches[1:]:

            self.x = np.concatenate((self.x, x_batch), axis=0)
            self.y = np.concatenate((self.y, y_batch))

    def _load_batch_json(self, batchFileName):

        # template for subclasses
        print('Load batch from json is not implemented in subclass')
        pass
        
class TrainLoader(DataLoader):

    def __init__(self, split, numBatches, dataDirectory):

        super().__init__(numBatches, dataDirectory=dataDirectory)
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.trainData = None
        self.valData = None
        self.split = split

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

        # normalize to center mean at zero
        self._normalize_instances()

        # split into train and test sets
        self._split_data()

        self.trainData = (self.x_train, self.y_train)
        self.valData = (self.x_val, self.y_val)

    def load(self, startBatch):

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
        
        return (self.x_train, self.y_train), (self.x_val, self.y_val)


