import json
import csv
import os

batchFileNames = os.listdir('./batches-train/')

for batchFileName in batchFileNames:

    rawData = {}

    with open('./batches-train/{}'.format(batchFileName), 'r') as f:
        rawData = json.load(f)

    with open('./batches-train-csv/{}.csv'.format(batchFileName[:-5]), 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',')

        for sampleNumber in range(len(rawData)):

            sample = rawData[str(sampleNumber)]

            csvWriter.writerow(sample['path']['x'])
            csvWriter.writerow(sample['path']['y'])
            csvWriter.writerow(sample['path']['theta'])
            csvWriter.writerow([sample['target']['index']])
