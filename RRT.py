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

    def __init__(self, dubinsCar, startingPosition, target):

        self.car = dubinsCar
        self.startingNode = self.NodeRRT(startingPosition) 
        self.target = target
        
    
def test_dubins_car_RRT():

    # set car original position
    startPosition = np.array([0.0, 0.0, 0.0])

    # set target
    target = np.array([10.0, 10.0])

    # configure and create dubins car
    velocity = 1.0
    maxSteeringAngle = (math.pi / 4.0) 
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
    dubinsCar = DubinsCar(startPosition, velocity, U)

    rrtSimulator = DubinsCarRRT(dubinsCar, startPosition, target)


if __name__ == '__main__':

    test_dubins_car_RRT()
        
    

    

