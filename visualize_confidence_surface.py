import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import argparse
import sys
from mpl_toolkits.mplot3d import Axes3D
from dubins_path_planner.scene import Scene
from data.mean_path_loader import MeanPathDataLoader

class SurfaceVisualizer:

    def __init__(self, target, paths, scatter=True, mesh=False, obs=False):

        self.target = target
        self.scene = Scene('test_room')
        self._setup_figure()
        self.paths = np.array(paths[:11])
        self.predictions = None
        self.scatter = scatter
        self.mesh = mesh
        self.obs = obs

    def _setup_figure(self):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_xlim(self.scene.dimensions['xmin'], self.scene.dimensions['xmax'])
        self.ax.set_ylim(self.scene.dimensions['ymin'], self.scene.dimensions['ymax'])
        self.ax.set_zlim(0.0, 1.0)
        self.ax.set_xlabel('x-position (m)')
        self.ax.set_ylabel('y-position (m)')
        self.ax.set_zlabel('GT-label confidence')
        self.ax.set_title('Classifier confidence w.r.t. X and Y Position')

    def _load_predictions(self):

        fileName = os.path.join('./predictions/', 'preds_{}.csv'.format(self.target))
        predictions = []

        with open(fileName, 'r') as f:

            reader = csv.reader(f, delimiter=';')
            for row in reader:

                if len(row) ==  0:
                    continue
                predictionSingleInstance = []
                for pred in row:
                    pred = pred.strip('[]')
                    pred = pred.split(',')
                    pred = [float(p) for p in pred]
                    predictionSingleInstance.append(pred)

                predictions.append(predictionSingleInstance)

        self.predictions = np.array(predictions)

    def _draw_cylinder(self, xCenter, yCenter, r, color='b'):

        maxZ = 1.0
        if color != 'b':
            maxZ = 0.2

        x = np.linspace(xCenter-r, xCenter+r, 100)
        z = np.linspace(0.0, maxZ, 100)
        xc, zc = np.meshgrid(x,z)
        yc = np.sqrt(r**2 -  (xc - xCenter)**2) + yCenter

        rstride = 20
        cstride = 10

        a = 0.0
        if color == 'b':
            a = 0.5
        else:
            a = 0.5 

        self.ax.plot_surface(xc, yc, zc, alpha=a, rstride=rstride, cstride=cstride, color=color)
        self.ax.plot_surface(xc, (2.0 * yCenter - yc), zc, alpha=a, rstride=rstride, cstride=cstride, color=color)

    def _plot_surface(self):

        x = []
        y = []
        z = []
        for i in range(self.predictions.shape[0]):

            x.extend(self.paths[i, 0, :].tolist())
            y.extend(self.paths[i, 1, :].tolist())
            z.extend(self.predictions[i, :, target].tolist())

        colorMap = plt.get_cmap('copper')
        if self.scatter:
            preds = self.ax.scatter(x, y, z, c=z, cmap = colorMap)
        if self.mesh:
            preds = self.ax.plot_trisurf(x, y, z, cmap=colorMap, alpha=0.8)

        t = self.scene.targets[target]
        self._draw_cylinder(t[0], t[1], t[2])

        if self.obs:
            for obs in self.scene.obstacles:
                self._draw_cylinder(obs[0], obs[1], obs[2], 'teal')

        self.fig.colorbar(preds, ax = self.ax, shrink = 0.5, aspect = 5)

        plt.show()

    def visualize(self):

        self._load_predictions()
        self._plot_surface()

def get_dataset(target, dataDir, algo, numBatches):

    dirToLoad = os.path.join(dataDir, '{}_batches_train'.format(algo))
    split = 1.0
    loader = MeanPathDataLoader(numBatches, dirToLoad)
    dataset = loader.load()

    return dataset

def visualize_confidence_surface(target, dataDir, algo, numBatches, scatter, mesh, obs):

    dataset = get_dataset(target, dataDir, algo, numBatches)
    surfaceVisualizer = SurfaceVisualizer(target, dataset.pathsByLabel[target], scatter, mesh, obs)
    surfaceVisualizer.visualize()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--directory', type=str, default = './data/batches-train')
    parser.add_argument('--algo', type=str, help='Planning algorithm', default='rrt')
    parser.add_argument('--batches', type=int, help='number of training batches to load', default=10)
    parser.add_argument('--scatter', action='store_true', default=False)
    parser.add_argument('--mesh', action='store_true', default=False)
    parser.add_argument('--obs', action='store_true', default=False)

    args = parser.parse_args()

    target = args.target
    dataDir = args.directory
    if dataDir == 'tower':
        dataDir = 'D:\\path_planning_data\\'

    algo = args.algo
    numBatches = args.batches
    scatter = args.scatter
    mesh = args.mesh
    obs = args.obs

    if not scatter and not mesh:
        print('Please specify --scatter or --mesh')
        sys.exit(2)

    visualize_confidence_surface(target, dataDir, algo, numBatches, scatter, mesh, obs)