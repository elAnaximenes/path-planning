import numpy as np
import os
import json


class CNNTrainDataLoader():

    def __init__(self, split, numBatches, dataDirectory, batchSize=512):

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

    def _split_data():
        pass

    def _transform_timeseries(self):
        pass

    def _normalize_instances(self):
        pass

    def _pad_instances(self):
        pass

    def _combine_batches(self):
        pass

    def _pre_process_data(self):
        pass

    def _load_batch_json(self, batchFileName):

        rawData = {}
        with open(os.path.join(self.dataDirectory, batchFileName), 'r') as f:
            rawData = json.load(f)

        instances = []
        labels = []

        for sampleNumber in range(len(rawData)):

            sample = rawData[str(sampleNumber)]

            x = sample['path

    def load(self, startBatch)

        self.batches = []

        print('loading data...')
        for i in range(startBatch, startBatch+self.numBatchesToLoad):

            batchFileName = 'test_room_batch_{}.json'.format(i)
            print(batchFileName)

            x_batch, y_batch = self._load_batch_json(batchFileName)
            self.batches.append((x_batch, y_batch))

            self.numBatchesToLoad -= 1
            if self.numBatchesToLoad == 0:
                break

        delf.pre_process_data()

        return (self.x_train, self.y_train), (self.x_val, self.y_val)
