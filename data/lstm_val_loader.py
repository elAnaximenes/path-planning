import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from .lstm_train_loader import LstmTrainDataLoader


class LstmValidateDataLoader(LstmTrainDataLoader):

    def __init__(self, numBatches, dataDirectory = './data/batches-validate/'):

        super().__init__(self, split=None, numBatches=numBatches, dataDirectory=dataDirectory)
        self.dataset = None

    def _pre_process_data(self)

    def load(self, startBatch):

        self.batches = []
        self.batchFileNames = os.listdir(self.dataDirectory)[startBatch: startBatch+self.numBatchesToLoad]
    
        for batchFileName in self.batchFileNames:
            
            x_batch, y_batch = self.load_batch_json(batchFileName)
            self.batches.append((x_batch, y_batch))
    
    self._pre_process_data()

    return self.dataset
