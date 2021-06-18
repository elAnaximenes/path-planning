import json 
import sys
import argparse
import dubins_path_planner.RRT
from run_RRT import run_RRT

def save_json_format(samplesInBatch, batchNum, sceneName):
    with open('./batches-new/{}_batch_{}.json'.format(sceneName, batchNum), 'w') as outFile:
        json.dump(samplesInBatch, outFile)
        
def save_batch(samplesInBatch, batchNum, sceneName, saveFormat):
    print('saving batch {}'.format(batchNum))
    if saveFormat == 'json':
        print('saveing json')
        save_json_format(samplesInBatch, batchNum, sceneName)

parser = argparse.ArgumentParser(description='Create batches of training data from RRT dubins planner.')

parser.add_argument('--batches', type=int, help='Number of batches to create.', default=1)
parser.add_argument('--batchsize', type=int, help='Number of paths generated per batch.', default=10)
parser.add_argument('--scene', type=str, help='Name of scene.', default='simple_room')
parser.add_argument('--format', type=str, help='Format of save file..', default='json')

args = parser.parse_args()

for batchNum in range(args.batches):

    samplesInBatch = {}
    for sampleNum in range(args.samples):

        sample = None 
        while sample is None:

            sample = run_RRT(animate=False, sceneName=args.scene)

        samplesInBatch['{}'.format(sampleNum)] = sample

    save_batch(samplesInBatch, batchNum, args.scene, args.format)

print('finished')



        

