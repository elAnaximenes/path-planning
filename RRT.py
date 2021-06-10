import math
import sys
import random
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
        #if(self.animate):
            #self._setup_animation()

    def _get_path_from_node_to_point(self, originNode, destinationPoint):

        dubinsState = np.array([originNode.x, originNode.y, originNode.theta])
        self.car.set_state(dubinsState)
        planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)
        path = planner.run()

        return path

    def _is_point_reachable(self, originNode, destinationPoint, path=None):

        # checking if target is reachable
        if path is None:
            path = self._get_path_from_node_to_point(originNode, destinationPoint)
        
        # check every point in path from origin to destination
        for x,y in zip(path['x'], path['y']):
            point = np.array([x,y])

            # invalid path if point collides with obstacle
            for obstacle in self.obstacleList:
                if abs(np.linalg.norm(point - obstacle[:2])) < obstacle[2]:
                    return False

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

        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.ax.set_title('Dubins Car RRT')
        self.ax.axis('equal')
        self.ax.set_ylabel('Y-distance(M)')
        self.ax.set_xlabel('X-distance(M)')
        self.ax.plot(self.root.x, self.root.y, 'x')
        plt.show()

    def _animate(self):

        fig = plt.figure()

        plt.plot(self.nodeList[0].x, self.nodeList[0].y, color='red', marker = 'x', markersize=10)
        for node in self.nodeList[1:]:
            plt.plot(node.x, node.y, color='black', marker = 'x', markersize=5)
            i = 0
            for x, y in zip(node.path['x'], node.path['y']):
                if i % 1000 == 0:
                    plt.plot(x, y, color='green', marker = '.', markersize = 2)
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
        
        targetIdx = 0
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

            # check for viable path from parent node to new point
            if startNode is not None and self._is_point_reachable(startNode, randomPoint, shortestPath):

                self._add_node(startNode, shortestPath)
                isTargetReachable = self._is_point_reachable(self.nodeList[-1], target)

        
        if iteration < self.maxIter:
            # finally, connect last node to target and add target to nodelist
            finalPathToTarget = self._get_path_from_node_to_point(self.nodeList[-1], target)
            self._add_node(self.nodeList[-1], finalPathToTarget)

        self._animate()

def load_scene():

    scene = None

    with open('./data/scene.json') as f:
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

def test_dubins_car_RRT(animate=False):

    # set car original position
    startPosition = np.array([0.0, 0.0, 0.0])

    # configure and create dubins car
    velocity = 1.0
    maxSteeringAngle = (math.pi / 4.0) 
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
    dubinsCar = DubinsCar(startPosition, velocity, U)

    # get scene information
    targetList, obstacleList = load_scene() 

    rrtSimulator = DubinsCarRRT(dubinsCar, startPosition, targetList, obstacleList, animate=animate)

    path = rrtSimulator.simulate()

if __name__ == '__main__':

    animate = False
    if len(sys.argv) > 1:
        animate = True

    test_dubins_car_RRT(animate)
        
    

    

