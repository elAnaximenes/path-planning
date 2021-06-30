import sys
import argparse
import csv
import json
import dubins_path_planner.RRT
from run_RRT import run_RRT
from run_optimal_RRT import run_optimal_RRT

def save_json_format(samplesInBatch, batchNum, sceneName):

    with open('./data/batches-new/{}_batch_{}.json'.format(sceneName, batchNum), 'w') as outFile:
        json.dump(samplesInBatch, outFile)

def save_csv_format(samplesInBatch, batchNum, sceneName):

    with open('./data/batches-new/{}_batch_{}.csv'.format(sceneName, batchNum), 'w') as outFile:

        writer = csv.writer(outFile, delimiter=',')

        for sample in samplesInBatch:
            print(sample)
            writer.writerow(samplesInBatch[sample]['path']['x'])
            writer.writerow(samplesInBatch[sample]['path']['y'])
            writer.writerow(samplesInBatch[sample]['path']['theta'])
            writer.writerow([samplesInBatch[sample]['target']['index']])
        
def save_batch(samplesInBatch, batchNum, sceneName, saveFormat):
    print('saving batch {}'.format(batchNum))
    if saveFormat == 'json':
        print('saving json')
        save_json_format(samplesInBatch, batchNum, sceneName)

    elif saveFormat.lower() == 'csv':
        print('saving csv')
        save_csv_format(samplesInBatch, batchNum, sceneName)

parser = argparse.ArgumentParser(description='Create batches of training data from RRT dubins planner.')

parser.add_argument('--batches', type=int, help='Number of batches to create.', default=1)
parser.add_argument('--batchsize', type=int, help='Number of paths generated per batch.', default=10)
parser.add_argument('--scene', type=str, help='Name of scene.', default='simple_room')
parser.add_argument('--format', type=str, help='Format of save file..', default='json')
parser.add_argument('--start_index', type=int, help='Index to begin saving batches at', default=0)
parser.add_argument('--algorithm', type=str, help='RRT or optimal', default = 'RRT')

args = parser.parse_args()

for batchNum in range(args.start_index, args.start_index + args.batches):

    samplesInBatch = {}
    for sampleNum in range(args.batchsize):

        sample = None 
        while sample is None:

            if args.algorithm == 'RRT':
                sample = run_RRT(animate=False, sceneName=args.scene)
            else:
                try:
                    sample = run_optimal_RRT(animate=False, sceneName=args.scene)
                except:
                    print('an exception occurred')
                    sample = None
                    continue

        samplesInBatch['{}'.format(sampleNum)] = sample

    save_batch(samplesInBatch, batchNum, args.scene, args.format)

print('finished')



        

