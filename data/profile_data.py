import os
import json
import numpy as np
import matplotlib.pyplot as plt
 
def histogram_path_lengths(pathLengths):
     
    plt.hist(pathLengths, bins=50) 

    mean, std = get_mean_and_std(pathLengths)

    plt.axvline(mean, ls="--", color='red')
    plt.axvline(mean+std, ls="--", color='yellow')
    plt.axvline(mean-std, ls="--", color='yellow')
    
    plt.xlabel('Path Lengths')
    plt.ylabel('Number of Paths')
    plt.title('Distribution of path lengths')

    plt.show()

    return mean, std

def get_mean_and_std(data):

    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std

def load_batch_json(batchFileName):

    # load raw json dict
    rawData = {}
    with open('./batches-train/{}'.format(batchFileName), 'r') as f:
        rawData = json.load(f)

    # build a list of instances and labels
    labels = [] 
    pathLengths = []

    print('sample length:', len(rawData))
    for sampleNumber in range(len(rawData)):

        sample = rawData[str(sampleNumber)]
        
        x = sample['path']['x']
        label = int(sample['target']['index'])

        pathLengths.append(len(x))
        labels.append(label)

    return pathLengths, labels

pathLengths = []
labels = []
batchFileNames = os.listdir('./batches-train/')

for batchFileName in batchFileNames:
    print(batchFileName)

    lengths, targets = load_batch_json(batchFileName)

    labels += targets 
    pathLengths += lengths

longestPath = 0
for pathL in pathLengths:
    if pathL > longestPath:
        longestPath = pathL

print('longestPath:', longestPath)

print('label length', len(labels))
print('lengths length', len(pathLengths))
mean, std = histogram_path_lengths(pathLengths)

print('Path length mean:', mean)
print('Path length std:', std)

targetCounts = {}

for label in labels:

    if label not in targetCounts:
        targetCounts[label] = 0

    targetCounts[label] += 1

plt.bar(targetCounts.keys(), targetCounts.values())

plt.title('Distribution of target classes')
plt.xlabel('Target Index')
plt.ylabel('Count in training set')
plt.show()
