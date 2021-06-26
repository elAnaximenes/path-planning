import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from .base_loaders.loaders import TrainLoader

class LstmTrainDataLoader(TrainLoader):

    def __init__(self, split, numBatches, dataDirectory):

        super().__init__(split, numBatches, dataDirectory=dataDirectory)
        
    def _pad_instances(self):

        """
        maxLen = 0
        for i in range(self.numSamples):
            if len(self.x[i][0]) > maxLen:
                maxLen = len(self.x[i][0])
        """
        maxLen = 9000

        print('longest path:', maxLen)
        padded_samples = np.zeros(shape = (self.numSamples, 3, maxLen))
        
        for i in range(self.numSamples):

            padded_samples[i][0][:len(self.x[i][0])] = self.x[i][0]
            padded_samples[i][1][:len(self.x[i][1])] = self.x[i][1]
            padded_samples[i][2][:len(self.x[i][2])] = self.x[i][2]

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

        # split into train and test sets
        self._split_data()

        self.trainData = (self.x_train, self.y_train)
        self.valData = (self.x_val, self.y_val)
        
    def _load_batch_json(self, batchFileName):

        # load raw json dict
        rawData = {}
        with open('{}/{}'.format(self.dataDirectory, batchFileName), 'r') as f:
            rawData = json.load(f)

        # build a list of instances and labels
        instances = [] 
        labels = [] 

        for sampleNumber in range(len(rawData)):

            sample = rawData[str(sampleNumber)]
            
            x = sample['path']['x']
            y = sample['path']['y']
            theta = sample['path']['theta']
            instance = np.array([x,y,theta])
            label = sample['target']['index']

            instances.append(instance)
            labels.append(label)

        x_batch = instances
        y_batch = labels

        return (x_batch, y_batch)
