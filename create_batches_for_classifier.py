import json 
import sys
import argparse
import dubins_path_planner.RRT

def save_json_format(samplesInBatch, batchNum, sceneName):
    with open('./paths/{}_batch_{}.json'.format(sceneName, batchNum), 'w') as outFile:
        print('saving samples')
        json.dump(samplesInBatch, outFile)
        
def save_batch(samplesInBatch, batchNum, sceneName, saveFormat):
    print('saving batch {}'.format(batchNum))
    if saveFormat == 'json':
        print('saveing json')
        save_json_format(samplesInBatch, batchNum, sceneName)

parser = argparse.ArgumentParser(description='Create batches of training data from RRT dubins planner.')

parser.add_argument('--batches', type=int, help='Number of batches to create.', default=1)
parser.add_argument('--samples', type=int, help='Number of paths generated per batch.', default=10)
parser.add_argument('--scene', type=str, help='Name of scene.', default='simple_room')
parser.add_argument('--format', type=str, help='Format of save file..', default='json')

args = parser.parse_args()

for batchNum in range(args.batches):

    samplesInBatch = {}
    for sampleNum in range(args.samples):

        path = None
        sample = {}
        while path is None:

            sample = RRT.test_dubins_car_RRT(animate=False, sceneName=args.scene)
            path = sample['path']

        print('finished sample {}'.format(sampleNum))
        samplesInBatch['{}'.format(sampleNum)] = sample

    save_batch(samplesInBatch, batchNum, args.scene, args.format)

print('finished')



        

