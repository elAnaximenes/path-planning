import math
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from car_models.dubins_optimal_planner import DubinsOptimalPlanner
from car_models.dubins_model import DubinsCar

class DubinsCarRRT:
    
    class NodeRRT:

        def __init__(self, position, state=None):

            self.x = position[0] 
            self.y = position[1] 
            self.theta = position[2] 
            self.path = {'x':[], 'y':[], 'theta':[]}
            self.parent = None

    def __init__(self, dubinsCar, startingPosition, targetList, obstacleList = None):

        self.car = dubinsCar
        self.root = self.NodeRRT(startingPosition) 
        self.nodeList = [self.root]
        self.targetList = targetList
        self.obstacleList = obstacleList

    def _is_point_reachable(self, originNode, destinationPoint, path=None):

        if path is None:
            dubinsState = np.array([originNode.x, originNode.y, originNode.theta])
            self.car.set_state(dubinsState)
            planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)
            path = planner.run()

        for x,y in zip(path['x'], path['y']):
            point = np.array([x,y])

            for obstacle in self.obstacleList:
                if abs(np.linalg.norm(point - obstacle[:2])) < obstacle[2]:
                    print('not a viable path')
                    return False
        print('found viable path')
        return True 

    def _calculate_dubins_path_length(self, originNode, destinationPoint):

        dubinsState = np.array([originNode.x, originNode.y, originNode.theta])
        self.car.set_state(dubinsState)

        planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)

        path = planner.run()
        pathLength = planner.angularDistanceTraveled + planner.linearDistanceTraveled

        return path, pathLength

    def simulate(self):
        
        targetIdx = 0
        # select target from list
        target = self.targetList[targetIdx]

        # check for valid path from root to target
        isTargetReachable = self._is_point_reachable(self.root, target)

        while not isTargetReachable:

            # sample random point
            randomPoint = np.random.uniform(low = -10.0, high = 10.0, size = (2,)) 

            # setup to begin search 
            shortestPath = None
            shortestPathLength = None
            startNode = None
            
            for node in self.nodeList:

                # get dubins optimal path and length
                path, pathLength = self._calculate_dubins_path_length(node, randomPoint)

                # store shortest path
                if shortestPathLength is None or pathLength < shortestPathLength:
                    shortestPathLength = pathLength
                    shortestPath = path
                    startNode = node

            # check for viable path from parent node to new point
            if self._is_point_reachable(startNode, randomPoint, shortestPath):
                nodeToAdd = NodeRRT(randomPoint)

def test_dubins_car_RRT():

    # set car original position
    startPosition = np.array([0.0, 0.0, 0.0])



    # configure and create dubins car
    velocity = 1.0
    maxSteeringAngle = (math.pi / 4.0) 
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
    dubinsCar = DubinsCar(startPosition, velocity, U)

    # set targets (x, y, radius)
    targetList = [[5.0, 5.0, 1.0]]
    targetList = [np.array(target) for target in targetList]

    # set obstacles (x, y, radius)
    obstacleList = [[2.5, 2.5, 1.0]]
    obstacleList = [np.array(obstacle) for obstacle in obstacleList]

    rrtSimulator = DubinsCarRRT(dubinsCar, startPosition, targetList, obstacleList)

    path = rrtSimulator.simulate()

if __name__ == '__main__':

    test_dubins_car_RRT()
        
    

    

