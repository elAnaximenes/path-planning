import numpy as np
import json

class DataLoader:

    def __init__(self):
        pass

    def _normalize_instances(self, x_train, x_val):

        mean = x_train.mean(axis=0, dtype=np.float64)
        x_train -= mean
        std = x_train.std(axis=0, dtype=np.float64)

        x_train /= std

        x_val -= mean
        x_val /= std

        return x_train, x_val

    def _combine_batches(self, batches):

        x = batches[0][0]
        y = batches[0][1]

        for x_batch, y_batch in batches[1:]:

            x = np.concatenate((x, x_batch), axis=0)
            y = np.concatenate((y, y_batch))

        return x, y

    def _split_data(self, x, y, trainValSplit):

        splitIndex = int(x.shape[0] * trainValSplit)
        x_train = x[:splitIndex]
        y_train = y[:splitIndex]
        x_val = x[splitIndex:]
        y_val = y[splitIndex:]

        return (x_train, y_train), (x_val, y_val)

    def _pre_process_data(self, batches, trainValSplit, minPathLength, maxPathLength):

        x = [] 
        y = []

        for x_batch, y_batch in batches:

            validX = []
            validY = []
            for instance, label in zip(x_batch, y_batch): 
                
                # throw out paths that are too long or too short
                pathLength = len(instance[0])
                if pathLength < minPathLength or pathLength > maxPathLength:
                    continue

                # truncate paths so that we have a fixed length input 
                instance[0] = instance[0][:minPathLength]
                instance[1] = instance[1][:minPathLength]
                instance[2] = instance[2][:minPathLength]

                # these are the paths we are going to use
                validX.append(instance)
                validY.append(label)

            x += validX
            y += validY 

        # transform labels to one hot
        x = np.array(x)
        y = to_categorical(np.array(y))

        # split into train and test sets
        (x_train, y_train), (x_val, y_val) = split_data(x, y, trainValSplit)

        # normalize to center mean at zero
        x_train, x_val = normalize_instances(x_train, x_val)

        return (x_train, y_train), (x_val, y_val)

    def _get_mean_and_std(self, data):

        data = np.array(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return mean, std

    def _load_batch(self, batchFileName):

        # load raw json dict
        rawData = {}
        with open('./batches-train/{}'.format(batchFileName), 'r') as f:
            rawData = json.load(f)

        # build a list of instances and labels
        instances = []
        labels = [] 
        batchPathLengths = []

        for sampleNumber in range(len(rawData)):

            sample = rawData[str(sampleNumber)]

            samplePathLength = len(sample['path']['x'])
            batchPathLengths.append(round(float(samplePathLength)/10.0) * 10)

            instance = [sample['path']['x'], sample['path']['y'], sample['path']['x']]
            instances.append(instance)
            labels.append(sample['target']['index'])
            
        x_batch = instances
        y_batch = labels

        return (x_batch, y_batch), batchPathLengths
    
    def load(self):
        pass

        


