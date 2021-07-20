import matplotlib.pyplot as plt
import json
import argparse
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import dubins_path_planner.RRT as RRT
from dubins_path_planner.scene import Scene

def draw_scene(scene, path, sampleNum):

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title('Test Room Path - {}'.format(sampleNum))
    plt.xlim(scene.dimensions['xmin'] - 1.0, scene.dimensions['xmax'] + 1.0)
    plt.ylim(scene.dimensions['ymin'] - 1.0, scene.dimensions['ymax'] + 1.0)

    ax.set_aspect('equal')
    ax.set_ylabel('Y-distance(M)')
    ax.set_xlabel('X-distance(M)')
    ax.plot(path['x'][0], path['y'][0], 'x')

    for obstacle in scene.obstacles:

        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        ax.add_patch(obs)
        fig.canvas.draw()

    for target in scene.targets:

        tar = plt.Circle((target[0], target[1]), target[2], color='blue', fill=False)
        ax.add_patch(tar)
        fig.canvas.draw()
    
    ax.plot(path['x'], path['y'], color='blue', linestyle='-', markersize=2)
        
    #plt.savefig('./path_figures/path-{}.png'.format(sampleNum))
    plt.show()

def load_json(sceneName, batchNum, dataDirectory):

    print('loading paths')
    
    with open('{}/{}_batch_{}.json'.format(dataDirectory,sceneName, batchNum), 'r') as f:
        paths = json.load(f)
    print('paths loaded')

    return paths

def load_csv(sceneName, batchNum, numSamples):

    pass
            
def load_paths(sceneName, batchNum, numSamples, saveFormat, dataDirectory):

    if saveFormat == 'csv':
        paths = load_csv(sceneName, batchNum, numSamples)

    elif saveFormat == 'json':
        paths = load_json(sceneName, batchNum, dataDirectory)

    return paths
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display batches of training data from RRT dubins planner.')

    parser.add_argument('--scene', type=str, help='Scene to display.', default='test_room')
    parser.add_argument('--batch', type=int, help='Batch Number to display.', default=0)
    parser.add_argument('--batchsize', type=int, help='Number of samples in batch to display.', default=1)
    parser.add_argument('--format', type=str, help='Batch file format.', default='json')
    parser.add_argument('--start_index', type=int, help='What index to begin saving batches at', default=0)
    parser.add_argument('--directory', type = str, default = './data/batches-train')
    parser.add_argument('--algo', type = str, default = 'RRT')

    args = parser.parse_args()

    sceneName = args.scene
    scene = Scene(sceneName)
    print('scene loaded')

    batchNum = args.batch 
    numSamples = args.batchsize
    saveFormat = args.format 
    startIndex = args.start_index
    algo = args.algo
    dataDirectory = args.directory
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\{}_batches_train'.format(algo)

    paths = load_paths(sceneName, batchNum, numSamples, saveFormat, dataDirectory)
    for i in range(numSamples):
        print('drawing path {}'.format(i))
        path = paths['{}'.format(i)]['path']
        if len(path['x']) == 0:
            continue
        draw_scene(scene, path, i)
