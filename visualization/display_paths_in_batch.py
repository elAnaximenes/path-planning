import matplotlib.pyplot as plt
import json
import argparse
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from dubins_path_planner.scene import Scene

def draw_path(scene, path, sampleNum, ax=None, save=False, color='blue'):

    if ax == None:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title('Test Room Path - {}'.format(sampleNum))
        ax = scene.draw(ax)

    ax.plot(path['x'], path['y'], color=color, linestyle='-', markersize=2)
    if save:
        plt.savefig('./path_figures/path-{}.png'.format(sampleNum))

def load_json(sceneName, batchNum, dataDirectory):

    print('loading paths')
    
    with open('{}/batch_{}.json'.format(dataDirectory, batchNum), 'r') as f:
        paths = json.load(f)
    print('paths loaded')

    return paths

            
def load_paths(sceneName, batchNum, numSamples, dataDirectory):

    paths = load_json(sceneName, batchNum, dataDirectory)

    return paths
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display batches of training data from RRT dubins planner.')

    parser.add_argument('--scene', type=str, help='Scene to display.', default='test_room')
    parser.add_argument('--batch', type=int, help='Batch Number to display.', default=0)
    parser.add_argument('--batchsize', type=int, help='Number of samples in batch to display.', default=1)
    parser.add_argument('--start_index', type=int, help='What index to begin saving batches at', default=0)
    parser.add_argument('--directory', type = str, default = '../data/batches_new')
    parser.add_argument('--algo', type = str, default = 'RRT')
    parser.add_argument('--all', action='store_true', default = False, help='Display all paths on the same axis')
    parser.add_argument('--save', action='store_true', default = False, help='Save figure to file')

    args = parser.parse_args()

    sceneName = args.scene
    scene = Scene(sceneName)
    print('scene loaded')

    batchNum = args.batch 
    numSamples = args.batchsize
    startIndex = args.start_index
    algo = args.algo
    allPathsAtOnce = args.all
    dataDirectory = args.directory
    saveImages = args.save
    if dataDirectory == 'tower':
        dataDirectory = 'D:\\path_planning_data\\{}_batches_train'.format(algo)

    paths = load_paths(sceneName, batchNum, numSamples, dataDirectory)

    if allPathsAtOnce:
        fig, ax = plt.subplots()
        ax.set_title('{} Paths'.format(sceneName))
        ax = scene.draw(ax)

        targetColors = {0:'blue', 1:'red', 2:'green', 3:'orange', 4:'purple' }

        for i in range(numSamples):
            sample = paths['{}'.format(i)]
            if len(sample['path']['x']) == 0:
                continue
            draw_path(scene, sample['path'], i, ax=ax, color=targetColors[sample['target']['index']])
        plt.show()
    else:
        for i in range(numSamples):
            path = paths['{}'.format(i)]['path']
            if len(path['x']) == 0:
                continue
            draw_path(scene, path, i, save=saveImages)
            plt.show()
