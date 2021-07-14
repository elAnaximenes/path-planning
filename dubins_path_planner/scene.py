import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.lines import Line2D
import os

class Scene:

    def __init__(self, sceneName=None):

        self.name = sceneName.replace('_', ' ')
        self.targets = []
        self.obstacles = []
        self.dimensions = {'xmin': 0.0, 'ymin': 0.0, 'xmax': 0.0, 'ymax': 0.0}
        self.carStart = None
        if self.name is not None:
            self.load_scene_from_json(sceneName)

    def load_scene_from_json(self, sceneName):

        jsonScene= None

        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        sceneFileName = os.path.join(currentDirectory, './scenes/{}.json'.format(sceneName))

        with open(sceneFileName) as f:
            jsonScene = json.load(f)

        targetsFromJson= jsonScene['targets']
        for target in targetsFromJson:

            t = [target['x'], target['y'], target['radius']]
            self.targets.append(t)

        obstaclesFromJson = jsonScene['obstacles']
        for obstacle in obstaclesFromJson:

            o = [obstacle['x'], obstacle['y'], obstacle['radius']]
            self.obstacles.append(o)

        self.carStart = np.array([jsonScene['car']['x'], jsonScene['car']['y'], jsonScene['car']['theta']])

        self.dimensions = jsonScene['dimensions']
    
    def _draw_obstacles(self, ax):

        for obstacle in self.obstacles:

            obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
            ax.add_patch(obs)

        return ax

    def _draw_targets(self, ax):

        for target in self.targets:

            tar = plt.Circle((target[0], target[1]), target[2], color='blue', fill=False)
            ax.add_patch(tar)

        return ax

    def draw(self, ax):

        ax.set_xlim(self.dimensions['xmin'] - 1.0, self.dimensions['xmax'] + 1.0)
        ax.set_ylim(self.dimensions['ymin'] - 1.0, self.dimensions['ymax'] + 1.0)

        ax.set_aspect('equal')
        ax.set_ylabel('Y-Position(M)')
        ax.set_xlabel('X-Position(M)')

        ax = self._draw_obstacles(ax)
        ax = self._draw_targets(ax)

        return ax


