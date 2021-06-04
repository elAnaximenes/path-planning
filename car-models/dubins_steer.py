import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import patches
from dubins_model import DubinsCar

"""
##############

Command Line:
python dubins_steer.py [train]

train is optional 
argument  if  you
want  to  specify 
a    test    case

##############
"""
# rho (euclidean distance -> origin to target)
# beta (angle -> direct path to target and final header)
# alpha (angle -> initial header and direct path to target)

class Steer:

    def __init__(self, dubinsCar, origin, target, acceptableError=0.01, maxIterations=100, targetHeader=False):

        self.dubinsCar = dubinsCar 
        self.state = origin
        self.target = target
        self.maxIterations = maxIterations
        self.iterations = 0
        self.targetHeader = targetHeader
        self.error = self._calculate_error()
        self.acceptableError = acceptableError
    
    def _calculate_error(self):

        rho = np.linalg.norm(self.target[:2] - self.state[:2])

        alpha = (math.atan2(self.target[1] - self.state[1], self.target[0] - self.state[0]) % (math.pi * 2.0)) - self.state[2] 
        if alpha > math.pi:
            alpha = math.pi - alpha
        if abs(alpha) > math.pi / 2.0:
            alpha *= ((math.pi / 2.0) / abs(alpha))

        # is there a header the robot needs to achieve at the final position? 
        if self.targetHeader:
            beta = -1.0 * (self.state[2] + alpha - self.target[2])
        else:
            beta = 0.0

        error = np.array([rho, alpha, beta])

        return error

    def update_error(self):

        self.error = self._calculate_error()

    def run(self):

        path = {'x': [], 'y': [], 'theta': []}

        # keep stepping until max iter or within certain radius of target
        while (self.error[0] > self.acceptableError) and (self.iterations < self.maxIterations):

            controlInput = self._controller()

            carNextState = self.dubinsCar.step(controlInput).values() # dict values
            self.state = np.fromiter(carNextState, dtype=float) # convert to numpy array
            self.update_error()

            path['x'].append(self.state[0])
            path['y'].append(self.state[1])
            path['theta'].append(self.state[2])

            self.iterations += 1

        return path

    def _controller(self):
    
        alpha = self.error[1]
        beta = self.error[2] 
        kAlpha = 8.0
        kBeta = -1.5
        omega = (kAlpha * alpha) + (kBeta * beta)
        controlInput = omega

        return controlInput 
            
def plot_path(path, origin, target, acceptableError, initialAlpha):

    # draw car path as arrows on plot 
    i = 0
    for x,y,theta in zip(path['x'], path['y'], path['theta']):
        if i % 100 == 0:
            plt.quiver(x, y, math.cos(theta), math.sin(theta)) 
        i += 1

    # text shifts
    xShift = 0.1
    yShift = -0.2

    # origin
    plt.plot(origin[0], origin[1], 'x', color='red', markersize=25)
    originStr = 'x: {:.2f}\ny: {:.2f}'.format(origin[0], origin[1])
    plt.text(origin[0] + xShift, origin[1] + yShift, originStr) 

    # target
    targetArea = plt.Circle((target[0], target[1]), acceptableError, color='blue', fill=False)
    plt.gca().add_patch(targetArea)
    targetStr = 'x: {:.2f}\ny: {:.2f}'.format(target[0], target[1])
    plt.text(target[0] + xShift, target[1] + yShift, targetStr) 

    # display
    plt.title('Car Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis("equal")
    plt.show()

def instantiate_car(origin=[0.0, 0.0, 0.0], velocity=0.5, maxSteeringAngle=0.25*math.pi):

    # control input range
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)] 
    dubinsCarModel = DubinsCar(origin, velocity, inputRange = U, dt=0.001)

    return dubinsCarModel

def simulate_steer_function(origin = np.array([0.0, 0.0, 0.0]), target = np.array([2.0, 2.0, 0.0]) ):

    # set up model
    acceptableError = 0.1
    maxIterations = 10000
    dubinsCar = instantiate_car(origin)
   
    # set up agent 
    steerFunction = Steer(dubinsCar, origin, target, acceptableError, maxIterations)
    
    # simulate path from origin to target
    path = steerFunction.run()
    plot_path(path, origin, target, acceptableError, steerFunction) 

if __name__ == '__main__':
    
    userSelection = 'TEST'

    if len(sys.argv) == 2:
        userSelection = sys.argv[1].upper()

    # for development, can specify specific cases
    if userSelection == 'TRAIN':
        origin = np.array([0.0, 0.0, 0.0])
        target = np.array([2.0, 2.0, 0.0])
        simulate_steer_function(origin, target)
        origin = np.array([0.0, 0.0, 0.0])
        target = np.array([2.0, -2.0, 0.0])
        simulate_steer_function(origin, target)
        origin = np.array([0.0, 0.0, 0.0])
        target = np.array([-2.0, -2.0, 0.0])
        simulate_steer_function(origin, target)
        origin = np.array([0.0, 0.0, 0.0])
        target = np.array([-2.0, 2.0, 0.0])
        simulate_steer_function(origin, target)

    # randomly generate test cases for steer function to handle
    else:
        for i in range(10):
            # set random origin
            origin = np.random.uniform(low = 0.0, high = 2.0, size = (3,)) 
            origin[2] = random.uniform(0.0, 2.0 * math.pi)

            # set random target
            target = np.random.uniform(low = 0.0, high = 2.0, size = (3,)) 
            target[2] = random.uniform(0.0, 2.0 * math.pi)

            # simulation
            simulate_steer_function(origin, target) 

