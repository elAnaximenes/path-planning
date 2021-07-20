import numpy as np
import json
import os
import tensorflow as tf
import math
from tensorflow.keras.utils import to_categorical
from .base_loaders.loaders import TrainLoader
import matplotlib.pyplot as plt

class LstmTrainDataLoader():

    def __init__(self, split, numBatches, dataDirectory, batchSize = 512, downSampleStride = 50):

        self.batchSize = batchSize
        self.downSampleStride = downSampleStride
        self.numBatchesToLoad = int(numBatches)
        self.dataDirectory = dataDirectory
        self.batches = None
        self.numSamples = 0
        self.x = None
        self.y = None
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

    def _transform_timeseries(self):

        X = self.x[:, 0, :]
        Y = self.x[:, 1, :]
        T = self.x[:, 2, :]

        # transform each instance from features, timesteps (n x t)
        # to timesteps, features (t x n)
        listX = []

        for i in range(X.shape[0]):

            timesteps = []

            for j in range(X.shape[1]):

                timesteps.append([X[i,j], Y[i,j], T[i,j]])

            listX.append(timesteps)
        self.x = np.array(listX) 
    
    def _normalize_instances(self):

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

    def _pad_instances(self):

        maxLen = 0
        for i in range(self.numSamples):
            if len(self.x[i][0]) > maxLen:
                maxLen = len(self.x[i][0])
        print('longest path:', maxLen)

        padded_samples = np.zeros(shape = (self.numSamples, 3, maxLen))
        
        # insert each sample into our numpy array of zeros
        for i in range(self.numSamples):

            broadcast = min(len(self.x[i][0]), maxLen)
            if broadcast == maxLen:
                padded_samples[i][0] = self.x[i][0][:maxLen]
                padded_samples[i][1] = self.x[i][1][:maxLen]
                padded_samples[i][2] = self.x[i][2][:maxLen]
            else:
                padded_samples[i][0][:broadcast] = self.x[i][0]
                padded_samples[i][1][:broadcast] = self.x[i][1]
                padded_samples[i][2][:broadcast] = self.x[i][2]

        self.x = padded_samples

    def _combine_batches(self):

        print('batches:',len(self.batches))
        print('batch size:',len(self.batches[0][0]))
        print('features:',len(self.batches[0][0][0]))

        self.x = self.batches[0][0]
        self.y = self.batches[0][1]

        for i in range(1, len(self.batches)):

            self.x.extend(self.batches[i][0])
            self.y.extend(self.batches[i][1])

        self.numSamples = len(self.x)

    def _pre_process_data(self):

        # combine batches into a rectangular tensor
        self._combine_batches()
        self._pad_instances()

        # transform labels to one hot
        self.y = to_categorical(np.array(self.y))

        # center mean at zero
        self._normalize_instances()

        self._transform_timeseries()

        # split into train and test sets
        self._split_data()

        self.trainData = (self.x_train, self.y_train)
        self.valData = (self.x_val, self.y_val)
        
    def _load_batch_json(self, batchFileName):

        # load raw json dict
        rawData = {}
        with open(os.path.join(self.dataDirectory, batchFileName), 'r') as f:
            rawData = json.load(f)

        # build a list of instances and labels
        instances = [] 
        labels = [] 

        for sampleNumber in range(len(rawData)):

            sample = rawData[str(sampleNumber)]
            
            x = sample['path']['x']
            y = sample['path']['y']
            theta = sample['path']['theta']

            downSampledX = []
            downSampledY = []
            downSampledTheta = []

            # DOWNSAMPLING
            for i in range(0, len(x), self.downSampleStride):
                downSampledX.append(x[i])
                downSampledY.append(y[i])
                downSampledTheta.append(theta[i])

            instance = np.array([downSampledX, downSampledY, downSampledTheta])
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
        
        return (self.x_train, self.y_train), (self.x_val, self.y_val)
