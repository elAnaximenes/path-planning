import json
import argparse
import os
import matplotlib.pyplot as plt
from dubins_path_planner.scene import Scene
from data.mean_path_loader import MeanPathDataLoader 

def plot_mean_paths(dataset, scene):

    meanPaths = dataset['mean paths']
    pathsByLabel = dataset['all paths']

    for label in meanPaths:

        fig, ax = plt.subplots()
        ax = scene.draw(ax)

        for i in range(100):#pathsByLabel[label].shape[0]):
            ax.plot(pathsByLabel[label][i, 0, :], pathsByLabel[label][i, 1, :], 'y')
        ax.plot(meanPaths[label][0, :], meanPaths[label][1, :], 'r--')

        plt.show()
        
def get_dataset(dataDir, algo, numBatches):

    dirToLoad = os.path.join(dataDir, '{}_batches_train'.format(algo))
    split = 1.0
    loader = MeanPathDataLoader(numBatches, dirToLoad)
    dataset = loader.load()

    return dataset

def get_mean_paths(dataDir, algo, numBatches):

    sceneName = 'test_room'
    scene = Scene(sceneName)
    dataset = get_dataset(dataDir, algo, numBatches)
    plot_mean_paths(dataset, scene)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--directory', type=str, default = './data/batches-train')
    parser.add_argument('--algo', type=str, help='Planning algorithm', default='rrt')
    parser.add_argument('--batches', type=int, help='number of training batches to load', default=10)

    args = parser.parse_args()

    dataDir = args.directory
    if dataDir == 'tower':
        dataDir = 'D:\\path_planning_data\\'

    algo = args.algo
    numBatches = args.batches

    get_mean_paths(dataDir, algo, numBatches)
