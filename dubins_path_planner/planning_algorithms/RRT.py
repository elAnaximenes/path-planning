import math
import os 
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from car_models.dubins_optimal_planner import DubinsOptimalPlanner
from car_models.dubins_model import DubinsCar
from matplotlib.lines import Line2D
from scene import Scene

"""
USAGE: python RRT.py [animate] [scene_selection]
eg: python RRT.py animate cluttered_room
"""

class DubinsCarRRT:
    
    class NodeRRT:

        def __init__(self, position, path=None):

            self.x = position[0] 
            self.y = position[1] 
            self.theta = position[2] 
            self.position = position
            self.parent = None
            self.path = {'x':[], 'y':[], 'theta':[]}
            if path is not None:
                self.path = path

    def __init__(self, dubinsCar, scene, animate = False):

        self.car = dubinsCar
        self.root = self.NodeRRT(scene.carStart) 
        self.nodeList = [self.root]
        self.scene = scene
        self.animate = animate
        self.fig = None
        self.ax = None
        self.maxIter = 100
        self.pathFromStartToTarget = None 
        self.leg=None
        if self.animate:
            self._setup_animation()
        self.text = None
        self.imgcount = 0

    def _select_random_target(self):

        targetIdx = random.randint(0, len(self.scene.targets) - 1)
        target = self.scene.targets[targetIdx]

        return target, targetIdx

    def _sample_random_point(self):

        randomPoint = np.zeros(2,)
        randomPoint[0] = random.uniform(self.scene.dimensions['xmin'], self.scene.dimensions['xmax']) 
        randomPoint[1] = random.uniform(self.scene.dimensions['ymin'], self.scene.dimensions['ymax']) 

        return randomPoint

    def _is_path_out_of_bounds(self, x, y):

        if x > self.scene.dimensions['xmax']\
                or x < self.scene.dimensions['xmin']\
                or y > self.scene.dimensions['ymax']\
                or y < self.scene.dimensions['ymin']:
                    return True
        return False

    def _is_point_reachable(self, originNode, destinationPoint, path=None):

        if originNode == None:
            return False

        # checking if target is reachable
        if path is None:
            path = self._get_path_from_node_to_point(originNode, destinationPoint)
            if path is None:
                return False
        
        i = 0
        # check for obstacle collisions
        for x,y in zip(path['x'], path['y']):
            point = np.array([x,y])

            if i % 10 == 0:

                if self._is_path_out_of_bounds(x, y):
                    return False

                # invalid path if point collides with obstacle
                for obstacle in self.scene.obstacles:
                    if abs(np.linalg.norm(point - obstacle[:2])) < obstacle[2]:
                        return False
            i += 1

        return True

    def _find_nearest_node_to_new_point(self, randomPoint):

        # setup to begin search 
        shortestPath = None
        shortestPathLength = None
        startNode = None
        
        # search tree for nearest neighbor to new point
        for node in self.nodeList:

            # ignore nodes that are too close to point 
            if abs(np.linalg.norm(node.position[:2] - randomPoint)) < (2.0 * self.car.minTurningRadius):
                continue

            # get dubins optimal path and length
            path, pathLength = self._calculate_dubins_path_length(node, randomPoint)

            # store shortest path
            if shortestPathLength is None or pathLength < shortestPathLength:
                shortestPathLength = pathLength
                shortestPath = path
                startNode = node

                if self.animate:
                    self._update_animation(point=randomPoint, path=shortestPath, event='candidate')

        return shortestPath, shortestPathLength, startNode

    def _get_path_from_node_to_point(self, startNode, destinationPoint):

        if abs(np.linalg.norm(startNode.position[:2] - destinationPoint[:2])) < (2.0 * self.car.minTurningRadius):
            return None 

        dubinsState = np.array([startNode.x, startNode.y, startNode.theta])
        self.car.set_state(dubinsState)
        planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)
        path = planner.run()

        return path

    def _calculate_dubins_path_length(self, originNode, destinationPoint):

        # re-initialize dubins car to be at origin node
        dubinsState = originNode.position 
        self.car.set_state(dubinsState)

        # instantiate optimal planner
        planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)

        # get optimal path produced by planner
        path = planner.run()
        pathLength = planner.angularDistanceTraveled + planner.linearDistanceTraveled

        return path, pathLength

    def _add_node(self, startNode, shortestPath):

        carStateAtPoint = np.array([shortestPath['x'][-1], shortestPath['y'][-1], shortestPath['theta'][-1]])
        nodeToAdd = self.NodeRRT(carStateAtPoint, shortestPath)
        nodeToAdd.parent = startNode
        nodeToAdd.path = shortestPath
        self.nodeList.append(nodeToAdd)

        return nodeToAdd

    def _set_final_path_from_start_to_target(self, target):

        finalPathToTarget = self._get_path_from_node_to_point(self.nodeList[-1], target)
        targetNode = self._add_node(self.nodeList[-1], finalPathToTarget)

        self.pathFromStartToTarget = {'nodes': [], 'path': {'x': [], 'y': [], 'theta': []}} 
        node = targetNode

        while node is not None:

            self.pathFromStartToTarget['nodes'].append(node)
            self.pathFromStartToTarget['path']['x'] = node.path['x'] + self.pathFromStartToTarget['path']['x']
            self.pathFromStartToTarget['path']['y'] = node.path['y'] + self.pathFromStartToTarget['path']['y']
            self.pathFromStartToTarget['path']['theta'] = node.path['theta'] + self.pathFromStartToTarget['path']['theta']
            node = node.parent

        if self.animate:
            self._update_animation(target, path=self.pathFromStartToTarget['path'], event='reached target') 
        
    def _draw_point_and_path(self, point, pathToPlot, colorSelection, style = '-' ):

        plottedPoint, = self.ax.plot(point[0], point[1], color='black', marker = 'x', markersize=5)
        plottedPath, = self.ax.plot(pathToPlot['x'], pathToPlot['y'], color=colorSelection,  linestyle=style, markersize = 2)

        return plottedPoint, plottedPath

    def _draw_obstacle(self, obstacle):

        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        self.ax.add_patch(obs)
        self.fig.canvas.draw()

    def _draw_target(self, target, colorSelection='blue'):

        tar = plt.Circle((target[0], target[1]), target[2], color=colorSelection, fill=False)
        self.ax.add_patch(tar)
        self.fig.canvas.draw()

    def _setup_animation(self):

        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.ax.set_title('Dubins Car RRT - {}'.format(self.scene.name))
        plt.xlim(self.scene.dimensions['xmin'] , self.scene.dimensions['xmax'] )
        plt.ylim(self.scene.dimensions['ymin'] , self.scene.dimensions['ymax'] )
        self.ax.set_aspect('equal')
        self.ax.set_ylabel('Y-distance(M)')
        self.ax.set_xlabel('X-distance(M)')
        self.ax.plot(self.root.x, self.root.y, 'x')

        legend_elements = [ Line2D([0], [0], marker='o', linestyle='', fillstyle='none', label='Non-selected Target',
                            markeredgecolor='blue', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Selected Target',
                            markeredgecolor='green', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Obstacle',
                            markeredgecolor='red', markersize=15)]

        for obstacle in self.scene.obstacles:

            self._draw_obstacle(obstacle)
            #plt.pause(0.0001)

        for target in self.scene.targets:

            self._draw_target(target)
            #plt.pause(0.0001)

        self.leg = self.ax.legend(handles=legend_elements, loc='best')


    def _write_caption(self, event):

        x = self.scene.dimensions['xmin'] + 0.5
        y = self.scene.dimensions['ymax'] - 1.0
        if event == 'candidate':
            self.text = plt.text(x, y, 'Calculating shortest path from tree to new point')
        elif event == 'valid path':
            self.text = plt.text(x, y, 'Path is possible without collisions')
        elif event == 'invalid path':
            self.text = plt.text(x, y, 'Path causes collision')
        elif event == 'reached target':
            self.text = plt.text(x, y, 'Path connects tree root to target')
    
    def _update_animation(self, point, path, event):
 
        if path is None:
            return 

        eventToColorDict = {'candidate': 'orange',\
                'valid path':'green',\
                'invalid path': 'red',\
                'reached target': 'blue'}

        pathToPlot = {'x': [], 'y':[]}
        i = 0
        for x, y in zip(path['x'], path['y']):

            if i % 1000 == 0:
                pathToPlot['x'].append(x)
                pathToPlot['y'].append(y)
            i += 1       

        plottedPoint, plottedPath = self._draw_point_and_path(point, path, eventToColorDict[event])
        
        self._write_caption(event)

        plt.pause(0.00001)

        #plt.savefig('./saved-images/rrt-{}.png'.format(self.imgcount))
        self.imgcount += 1

        if event != 'reached target':
            self.text.remove()

        if event == 'invalid path' or event == 'candidate':
            plottedPoint.remove()
            plottedPath.remove()

    def _extend(self, target):
 
        isTargetReachable = False

        randomPoint = self._sample_random_point()
        shortestPath, shortestPathLength, startNode = self._find_nearest_node_to_new_point(randomPoint)

        # check for viable path from parent node to new point
        isPointReachable = self._is_point_reachable(startNode, randomPoint, shortestPath)

        if isPointReachable:
            self._add_node(startNode, shortestPath)
            isTargetReachable = self._is_point_reachable(self.nodeList[-1], target)

            if self.animate:
                self._update_animation(point=randomPoint, path=shortestPath, event='valid path')

        elif self.animate:
            self._update_animation(point=randomPoint, path=shortestPath, event='invalid path')

        return isTargetReachable
    
       
    # RRT ALGORITHM
    def simulate(self):
        
        target, targetIdx = self._select_random_target()

        if self.animate:
            self._draw_target(target, 'lime')
            plt.pause(2.0)
            self.leg.remove()
        
        # check for valid path from root to target
        isTargetReachable = self._is_point_reachable(self.root, target)

        iteration = 0
        while not isTargetReachable and iteration < self.maxIter:

            # extend tree
            isTargetReachable = self._extend(target)
            iteration += 1

        sample = None
        # finally, connect last node to target and add target to nodelist
        if iteration < self.maxIter:
            self._set_final_path_from_start_to_target(target)
            sample = {}
            sample['target'] = {'coordinates': target, 'index': targetIdx }
            sample['path'] = self.pathFromStartToTarget['path']

        if self.animate:
            legend_elements = [ Line2D([0], [0], marker='o', linestyle='', fillstyle='none', label='Non-selected Target',
                            markeredgecolor='blue', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Selected Target',
                            markeredgecolor='green', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Obstacle',
                            markeredgecolor='red', markersize=15),
                            Line2D([0], [0], marker='_', fillstyle='none', linestyle='', label='RRT Tree Branches',
                            markeredgecolor='green', markersize=15),
                            Line2D([0], [0], marker='_', fillstyle='none', linestyle='', label='Final Path',
                            markeredgecolor='blue', markersize=15)
                            ]
            leg = self.ax.legend(handles=legend_elements, loc='best')
            plt.show()


        return sample


