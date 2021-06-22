import numpy as np
import json
import os
from tensorflow.keras.utils import to_categorical
from .loader.loader import DataLoader

class LstmDataLoader(DataLoader):

    def __init__(self, split, numBatches):

        super().__init__(split, numBatches)
    
    def _combine_batches(self):

        print('batches:',len(self.batches))
        print('batch size:',len(self.batches[0][0]))
        print('features:',len(self.batches[0][0][0]))
        exit(1)

    def _pre_process_data(self):

        self._combine_batches()

    def _load_batch_json(self, batchFileName):

        # load raw json dict
        rawData = {}
        with open('./data/batches-train/{}'.format(batchFileName), 'r') as f:
            rawData = json.load(f)

        # build a list of instances and labels
        instances = [] 
        labels = [] 

        for sampleNumber in range(len(rawData)):

            sample = rawData[str(sampleNumber)]
            
            x = sample['path']['x']
            y = sample['path']['y']
            theta = sample['path']['theta']

            """
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
            """

            instance = np.array([x,y,theta])
            label = sample['target']['index']

            instances.append(instance)
            labels.append(label)

        x_batch = instances
        y_batch = np.array(labels)

        return (x_batch, y_batch)
   
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

        


