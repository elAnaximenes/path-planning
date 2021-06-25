import numpy as np
import json
import os
from tensorflow.keras.utils import to_categorical
from .base_loaders.loaders import TrainLoader

class FeedForwardTrainDataLoader(TrainLoader):

    def __init__(self, split, numBatches, truncatedPathLength=1000, dataDirectory='./data/batches-train/'):

        super().__init__(split, numBatches, dataDirectory=dataDirectory)
        self.truncatedPathLength = truncatedPathLength 

    def _combine_batches(self):

        self.x = self.batches[0][0]
        self.y = self.batches[0][1]

        for x_batch, y_batch in self.batches[1:]:

            self.x = np.concatenate((self.x, x_batch), axis=0)
            self.y = np.concatenate((self.y, y_batch))

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

            if len(x) < self.truncatedPathLength:
                instance = np.zeros((3, self.truncatedPathLength))
                x = np.array(x)
                y = np.array(y)
                theta = np.array(theta)
                instance[0, :x.shape[0]] = x
                instance[0, :y.shape[0]] = y
                instance[0, :theta.shape[0]] = theta
            else:
                instance = np.array([\
                                    x[:self.truncatedPathLength],\
                                    y[:self.truncatedPathLength],\
                                    theta[:self.truncatedPathLength]\
                                    ])

            label = sample['target']['index']

            instances.append(instance)
            labels.append(label)

        x_batch = np.array(instances)
        y_batch = np.array(labels)

        return (x_batch, y_batch)
   
