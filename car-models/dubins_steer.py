import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import patches
from dubins_model import DubinsCar

# rho (euclidean distance -> origin to target)
# beta (angle -> direct path to target and final header)
# alpha (angle -> initial header and direct path to target)

def controller(dubinsCar, error):
    
    alpha = error[1]
    beta = error[2]
    kAlpha = 8.0
    kBeta = -1.5
    omega = (kAlpha * alpha) + (kBeta * beta)
    controlInput = omega

    return controlInput 

class Steer:

    def __init__(self, dubinsCar, origin, target, acceptableError=0.01, maxIterations=100):

        self.dubinsCar = dubinsCar 
        self.state = origin
        self.target = target
        self.error = self._calculate_error()
        self.acceptableError = acceptableError
        self.maxIterations = maxIterations
        self.iterations = 0
    
    def _calculate_error(self):

        error = np.zeros(3)
        # rho
        error[0] = np.linalg.norm(self.target[:2] - self.state[:2])
        # alpha
        error[1] = (math.atan2(self.target[1] - self.state[1], self.target[0] - self.state[0]) % (math.pi * 2.0)) - self.state[2] 
        # beta
        error[2] = -1.0 * (self.dubinsCar.state['theta'] + error[1] - self.target[2])

        return error

    def update_error(self):

        self.error = self._calculate_error()

    def run(self):

        path = {'x': [], 'y': [], 'theta': []}
        # keep stepping until max iter or within certain radius of target
        while (self.error[0] > self.acceptableError) and (self.iterations < self.maxIterations):
            controlInput = controller(self.dubinsCar, self.error)
            carNextState = self.dubinsCar.step(controlInput).values() # dict values
            self.state = np.fromiter(carNextState, dtype=float) # convert to numpy array
            self.update_error()
            path['x'].append(self.state[0])
            path['y'].append(self.state[1])
            path['theta'].append(self.state[2])
            self.iterations += 1

        return path
            
def plot_path(path, origin, target):

    i = 0
    # draw car path as arrows on plot 
    for x,y,theta in zip(path['x'], path['y'], path['theta']):
        if i % 100 == 0:
            plt.quiver(x, y, math.cos(theta), math.sin(theta)) 
        i += 1

    #plt.plot(path['x'], path['y'], '-', color='black')  # car path
    plt.plot(origin[0], origin[1], 'x', color='red', markersize=25) # car origin

    plt.plot(target[0], target[1], 'o', color='blue', markersize=25) # car origin
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
   
    # simulator
    steerFunction = Steer(dubinsCar, origin, target, acceptableError, maxIterations)
    
    # simulate path from origin to target
    path = steerFunction.run()
    plot_path(path, origin, target) 

if __name__ == '__main__':
    
    userSelection = 'TEST'
    if len(sys.argv) == 2:
        userSelection = sys.argv[1].upper()

    if userSelection == 'TRAIN':
        origin = np.array([0.0, 0.0, 0.0])
        target = np.array([-2.0, 2.0, -0.5 * math.pi])
        simulate_steer_function(origin, target)

    else:
        for i in range(10):
            origin = np.random.uniform(low = 0.0, high = 2.0, size = (3,)) 
            origin[2] = random.uniform(0.0, 2.0 * math.pi)
            target = np.random.uniform(low = 0.0, high = 2.0, size = (3,)) 
            target[2] = random.uniform(0.0, 2.0 * math.pi)
            simulate_steer_function(origin, target) 

