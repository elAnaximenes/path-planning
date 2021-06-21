import os
import json
from train_classifier import histogram_path_lengths
import matplotlib.pyplot as plt

def load_batch_json(batchFileName):

    # load raw json dict
    rawData = {}
    with open('./batches-train/{}'.format(batchFileName), 'r') as f:
        rawData = json.load(f)

    # build a list of instances and labels
    labels = [] 
    pathLengths = []

    for sampleNumber in range(len(rawData)):

        sample = rawData[str(sampleNumber)]
        
        x = sample['path']['x']
        label = int(sample['target']['index'])

        pathLengths.append(len(x))
        labels.append(label)

    return pathLengths, labels


pathLengths = []
labels = []
batchFileNames = os.listdir('./batches-train')

for batchFileName in batchFileNames:

    lengths, labels = load_batch_json(batchFileName)

    labels += labels
    pathLengths += lengths


mean, std = histogram_path_lengths(pathLengths)

print('Path length mean:', mean)
print('Path length std:', std)

targets = [0,1,2,3,4]
counts = [0,0,0,0,0]

for label in labels:

    counts[label] += 1

plt.bar(targets,counts)




    






