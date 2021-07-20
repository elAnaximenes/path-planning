import os
import json
import csv
import numpy as np
import argparse

def load_predictions(fileName):

    pass

def load_gradients(fileName):

    with open(fileName, 'r') as f:

        reader = csv.reader(f, delimiter=';')
        for row in reader:

            if len(row) == 0:
                continue

            gradientsSingleInstance = []
            for grad in row:

                grad = grad.replace('[', '')
                grad = grad.replace(']', '')
                grad = grad.split(',')
                grad = [float(g) for g in grad]
                grad = grad[3*self.target:3*(self.target+1)]

                gradientsSingleInstance.append(grad)
            
            print(len(gradientsSingleInstance))
            gradients.append(gradientsSingleInstance)

    return gradients

def get_lookup_source(modelSelection, source):

    fileNames = ['./{}/{}/tar_{}.csv'.format(source, modelSelection, target) for target in range(5)] 
    sourceData = {0:[], 1:[], 2:[], 3:[], 4:[]}

    for target in range(2):

        fileName = fileNames[target]

        if source == "gradients":
            sourceData[target] = load_gradients(fileName)
        elif source == "predictions":
            sourceData[target] = load_predictions(fileName)

    return sourceData

def create_lookup_table(modelSelection, source):

    lookupTableSourceData = get_lookup_source(modelSelection, source)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--source', type=str, default='gradients', help='gradients/confidence')
    
    args = parser.parse_args()

    modelSelection = args.model
    source = args.source

    create_lookup_table(modelSelection, source)
