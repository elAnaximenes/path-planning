import math
import sys
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from car_models.dubins_optimal_planner import DubinsOptimalPlanner
from car_models.dubins_model import DubinsCar

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


    def __init__(self, dubinsCar, startingPosition, targetList, obstacleList = None, animate = False):

        self.car = dubinsCar
        self.root = self.NodeRRT(startingPosition) 
        self.nodeList = [self.root]
        self.targetList = targetList
        self.obstacleList = obstacleList
        self.animate = animate
        self.fig = None
        self.ax = None
        self.maxIter = 100
        self.pathFromStartToTarget = {'nodes': [], 'path': {'x': [], 'y': [], 'theta': []}} 
        if(self.animate):
            self._setup_animation()

    def _get_path_from_node_to_point(self, originNode, destinationPoint):

        dubinsState = np.array([originNode.x, originNode.y, originNode.theta])
        self.car.set_state(dubinsState)
        planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)
        path = planner.run()

        return path

    def _is_point_reachable(self, originNode, destinationPoint, path=None):

        if originNode == None:
            return False

        # checking if target is reachable
        if path is None:
            path = self._get_path_from_node_to_point(originNode, destinationPoint)
        
        i = 0
        # check for obstacle collisions
        for x,y in zip(path['x'], path['y']):
            point = np.array([x,y])

            if i % 100 == 0:

                # invalid path if point collides with obstacle
                for obstacle in self.obstacleList:
                    if abs(np.linalg.norm(point - obstacle[:2])) < obstacle[2]:
                        return False

            i += 1

        return True

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
    
    def _setup_animation(self):

        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.ax.set_title('Dubins Car RRT')
        plt.xlim(-12.0, 12.0)
        plt.ylim(-12.0, 12.0)
        self.ax.set_aspect('equal')
        self.ax.set_ylabel('Y-distance(M)')
        self.ax.set_xlabel('X-distance(M)')
        self.ax.plot(self.root.x, self.root.y, 'x')

        for obstacle in self.obstacleList:

            obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
            self.ax.add_patch(obs)
            self.fig.canvas.draw()
            plt.pause(0.0001)

        for target in self.targetList:

            tar = plt.Circle((target[0], target[1]), target[2], color='blue', fill=False)
            self.ax.add_patch(tar)
            self.fig.canvas.draw()
            plt.pause(0.0001)
    
    def _update_animation(self, point, path, event):
 
        if path is None:
            return 

        pathToPlot = {'x': [], 'y':[]}

        i = 0
        for x, y in zip(path['x'], path['y']):
            sparsenessOfPath = 1000

            if i % sparsenessOfPath == 0:
                pathToPlot['x'].append(x)
                pathToPlot['y'].append(y)
            i += 1       

        if event == 'candidate':
            plottedPoint, = self.ax.plot(point[0], point[1], color='black', marker = 'x', markersize=5)
            plottedPath, = self.ax.plot(pathToPlot['x'], pathToPlot['y'], color = 'orange',  linestyle='dotted', markersize = 2)
            plt.pause(0.1)
            
            plottedPoint.remove()
            plottedPath.remove()

        else:
            if event == 'valid path':
                colorSelection = 'green'

            elif event == 'invalid path':
                colorSelection = 'red'

            elif event == 'reached target':
                colorSelection = 'blue'

            plottedPoint, = self.ax.plot(point[0], point[1], color='black', marker = 'x', markersize=5)
            plottedPath, = self.ax.plot(pathToPlot['x'], pathToPlot['y'], color=colorSelection, marker='.', markersize=2)
            plt.pause(0.1)

            if event == 'invalid path':
                plottedPath.remove()
                plottedPoint.remove()
        
    def _draw_static(self):

        fig = plt.figure()
        # root
        plt.plot(self.nodeList[0].x, self.nodeList[0].y, color='red', marker = 'x', markersize=10)

        for node in self.nodeList[1:]:

            plt.plot(node.x, node.y, color='black', marker = 'x', markersize=5)

            i = 0
            for x, y in zip(node.path['x'], node.path['y']):

                if i % 1000 == 0:
                    plt.plot(x, y, color='green', marker = '.', markersize = 2)
                i += 1

        i = 0
        for x, y in zip(self.pathFromStartToTarget['path']['x'], self.pathFromStartToTarget['path']['y']):

            if i % 1000 == 0:
                plt.plot(x, y, color='blue', marker = '.', markersize = 2)
            i += 1

        for obstacle in self.obstacleList:

            obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
            plt.gca().add_patch(obs)

        for target in self.targetList:

            tar = plt.Circle((target[0], target[1]), target[2], color='blue', fill=False)
            plt.gca().add_patch(tar)

        plt.axis("equal")
        plt.show()

    # RRT ALGORITHM
    def simulate(self):
        
        targetIdx = random.randint(0, len(self.targetList) - 1)
        # select target from list
        target = self.targetList[targetIdx]

        # check for valid path from root to target
        isTargetReachable = self._is_point_reachable(self.root, target)

        iteration = 0

        while not isTargetReachable and iteration < self.maxIter:

            iteration += 1

            # sample random point
            randomPoint = np.random.uniform(low = -10.0, high = 10.0, size = (2,)) 

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

            isPointReachable = self._is_point_reachable(startNode, randomPoint, shortestPath)
            # check for viable path from parent node to new point
            if isPointReachable:
                self._add_node(startNode, shortestPath)
                isTargetReachable = self._is_point_reachable(self.nodeList[-1], target)

                if self.animate:
                    self._update_animation(point=randomPoint, path=shortestPath, event='valid path')

            elif self.animate and not isPointReachable:
                self._update_animation(point=randomPoint, path=shortestPath, event='invalid path')
        
        if iteration < self.maxIter:
            # finally, connect last node to target and add target to nodelist
            finalPathToTarget = self._get_path_from_node_to_point(self.nodeList[-1], target)
            targetNode = self._add_node(self.nodeList[-1], finalPathToTarget)

            node = targetNode

            while node is not None:

                self.pathFromStartToTarget['nodes'].append(node)
                self.pathFromStartToTarget['path']['x'] = node.path['x'] + self.pathFromStartToTarget['path']['x']
                self.pathFromStartToTarget['path']['y'] = node.path['y'] + self.pathFromStartToTarget['path']['y']
                self.pathFromStartToTarget['path']['theta'] = node.path['theta'] + self.pathFromStartToTarget['path']['theta']
                node = node.parent

            if self.animate:
                self._update_animation(target, path=self.pathFromStartToTarget['path'], event='reached target') 

        if self.animate:
            plt.show()
        else:
            self._draw_static()

def load_scene(sceneSelection):

    scene = None

    with open('./scenes/{}.json'.format(sceneSelection)) as f:
        scene = json.load(f)

    targets = scene['targets']
    targetList = []
    for target in targets:

        t = [target['x'], target['y'], target['radius']]
        targetList.append(t)

    obstacles = scene['obstacles']
    obstacleList = []
    for obstacle in obstacles:

        o = [obstacle['x'], obstacle['y'], obstacle['radius']]
        obstacleList.append(o)
    
    return targetList, obstacleList

def test_dubins_car_RRT(sceneSelection, animate=False):

    # set car original position
    startPosition = np.array([0.0, 0.0, 0.0])

    # configure and create dubins car
    velocity = 1.0
    maxSteeringAngle = (math.pi / 4.0) 
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
    timeStep = 0.0001
    dubinsCar = DubinsCar(startPosition, velocity, U, dt=timeStep)

    # get scene information
    targetList, obstacleList = load_scene() 

    rrtSimulator = DubinsCarRRT(dubinsCar, startPosition, targetList, obstacleList, animate=animate)

    path = rrtSimulator.simulate()

if __name__ == '__main__':

    print('usage: {} scene_name [animate]'.format(sys.argv[0]))
    
    sceneSelection = 'cluttered_room'

    if len(sys.argv == 1):
        sceneSelection = sys.argv[1]

    animate = False
    if len(sys.argv) > 2:
        animate = True

    test_dubins_car_RRT(sceneSelection, animate)
        
    

    

