import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import argparse
import math
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from dubins_path_planner.scene import Scene
from data.mean_path_loader import MeanPathDataLoader

class GradientVisualizer:

    def __init__(self, target, paths, scatter=True, mesh=False, obs=False):

        self.target = target
        self.scene = Scene('test_room')
        self._setup_figure()
        self.paths = np.array(paths)
        self.gradients = None
        self.scatter = scatter
        self.mesh = mesh
        self.obs = obs

    def _setup_figure(self):

        self.fig = plt.figure()
        self.x_ax = self.fig.add_subplot(1,2,1, projection='3d')

        self.x_ax.set_xlim(self.scene.dimensions['xmin'], self.scene.dimensions['xmax'])
        self.x_ax.set_ylim(self.scene.dimensions['ymin'], self.scene.dimensions['ymax'])
        self.x_ax.set_zlim(-1.0, 1.0)

        self.x_ax.set_xlabel('x-position (m)')
        self.x_ax.set_ylabel('y-position (m)')
        self.x_ax.set_zlabel('GT-label confidence derivative w.r.t. X')
        self.x_ax.set_title('Classifier confidence derivative w.r.t. X Position')

        self.y_ax = self.fig.add_subplot(1,2,2, projection='3d')

        self.y_ax.set_xlim(self.scene.dimensions['xmin'], self.scene.dimensions['xmax'])
        self.y_ax.set_ylim(self.scene.dimensions['ymin'], self.scene.dimensions['ymax'])
        self.y_ax.set_zlim(-1.0, 1.0)

        self.y_ax.set_xlabel('x-position (m)')
        self.y_ax.set_ylabel('y-position (m)')
        self.y_ax.set_zlabel('GT-label confidence derivative w.r.t. Y')
        self.y_ax.set_title('Classifier confidence derivative w.r.t. Y Position')

    def _load_gradients(self):

        fileName = os.path.join('./gradients/', 'grads_{}.csv'.format(self.target))
        gradients = []

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

                gradients.append(gradientsSingleInstance)
        
        self.gradients = np.array(gradients)

    def _draw_cylinder(self, ax, xCenter, yCenter, r, color='b'):

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
            a = None 

        ax.plot_surface(xc, yc, zc, alpha=a, rstride=rstride, cstride=cstride, color=color)
        ax.plot_surface(xc, (2.0 * yCenter - yc), zc, alpha=a, rstride=rstride, cstride=cstride, color=color)

    def _compute_grid(self, index):

        gridSize = 20 

        x = np.linspace(-10.0, 10.0, gridSize)
        y = np.linspace(-10.0, 10.0, gridSize)

        z = np.zeros((gridSize, gridSize)) 
        count = np.ones((gridSize, gridSize))

        for i in range(self.paths.shape[0]-1):

            for j in range(self.paths.shape[2]-1):

                gridRow = math.floor((gridSize * (self.paths[i, 0, j] + 10.0)) / 20.0)
                gridCol = math.floor((gridSize * (self.paths[i, 1, j] + 10.0)) / 20.0)
                z[gridRow, gridCol] += self.gradients[i, j, index]
                count[gridRow, gridCol] += 1.0

        z /= count

        return x, y, z

    def _plot_surface(self, ax, index):

        x = []
        y = []
        z = []

        for i in range(self.gradients.shape[0]):

            x.extend(self.paths[i, 0, :-1].tolist())
            y.extend(self.paths[i, 1, :-1].tolist())
            z.extend(self.gradients[i, :, index].tolist())
        #x, y, z = self._compute_grid(index)


        colorMap = plt.get_cmap('Spectral')
        if self.scatter:
            preds = ax.scatter(x, y, z, c=z, cmap = colorMap)
        if self.mesh:
            preds = ax.plot_trisurf(x, y, z, cmap=colorMap, alpha=0.8)


        t = self.scene.targets[target]
        self._draw_cylinder(ax, t[0], t[1], t[2])

        if self.obs:
            for obs in self.scene.obstacles:
                self._draw_cylinder(ax, obs[0], obs[1], obs[2], 'teal')

        x, y = np.meshgrid(x, y)
        preds = ax.plot_surface(x,y,z, cmap = cm.Spectral, alpha = 0.9)

        self.fig.colorbar(preds, ax = ax, shrink = 0.5, aspect = 5)

    def visualize(self):

        self._load_gradients()
        self._plot_surface(self.y_ax, 0)
        self._plot_surface(self.x_ax, 1)
        plt.show()

class ConfidenceVisualizer:

    def __init__(self, target, paths, scatter=True, mesh=False, obs=False):

        self.target = target
        self.scene = Scene('test_room')
        self._setup_figure()
        self.paths = np.array(paths)
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
        print(self.predictions.shape)
        print(self.paths.shape)
        for i in range(self.predictions.shape[0]):

            x.extend(self.paths[i, 0, ].tolist())
            y.extend(self.paths[i, 1, ].tolist())
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

        self.fig.colorbar(preds, ax = self.ax, shrink = 0.3, aspect = 5)

        plt.show()

    def visualize(self):

        self._load_predictions()
        self._plot_surface()

def get_dataset(target, dataDir, algo, numBatches):

    """
    dirToLoad = os.path.join(dataDir, '{}_batches_train'.format(algo))
    split = 1.0
    loader = MeanPathDataLoader(numBatches, dirToLoad)
    dataset = loader.load()
    """
    valDataDir = os.path.join(dataDir, '{}_batches_train'.format(algo))
    stepSize = 100
    loader = ValidateDataLoader(numBatches, valDataDir, stepSize)
    dataset = loader.load()

    return dataset

def visualize_network_surface(target, dataDir, algo, numBatches, scatter, mesh, obs, surface='predictions'):

    dataset = get_dataset(target, dataDir, algo, numBatches)
    if surface == 'predictions':
        surfaceVisualizer = ConfidenceVisualizer(target, dataset.pathsByLabel[target], scatter, mesh, obs)
    elif surface == 'gradients':
        surfaceVisualizer = GradientVisualizer(target, dataset.pathsByLabel[target], scatter, mesh, obs)
    surfaceVisualizer.visualize()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--directory', type=str, default = './data/batches-train')
    parser.add_argument('--algo', type=str, help='Planning algorithm', default='optimal_rrt')
    parser.add_argument('--batches', type=int, help='number of training batches to load', default=10)
    parser.add_argument('--surface', type=str, help='predictions/gradients', default='predictions')
    parser.add_argument('--scatter', action='store_true', default=False)
    parser.add_argument('--mesh', action='store_true', default=False)
    parser.add_argument('--obs', action='store_true', default=False)

    args = parser.parse_args()
    print(args, flush=True)

    target = args.target
    dataDir = args.directory
    if dataDir == 'tower':
        dataDir = 'D:\\path_planning_data\\'

    algo = args.algo
    numBatches = args.batches
    scatter = args.scatter
    mesh = args.mesh
    obs = args.obs
    surface = args.surface

    if not scatter and not mesh:
        print('Please specify --scatter or --mesh')
        sys.exit(2)

    visualize_network_surface(target, dataDir, algo, numBatches, scatter, mesh, obs, surface)
