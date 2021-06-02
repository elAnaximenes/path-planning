# See planning.cs.uiuc.edu chapter 15.3.1(Dubins Curves) for reference material
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# helper function keeps control variable in allowable range
def clip(value, minimum, maximum):

    if value > maximum:
        value = maximum
    elif value < minimum:
        value = minimum
    return value

def get_num_steps(dubinsPrimitive):
    numSteps = 0
    if dubinsPrimitive == 'S':
        numSteps = 1000
    elif dubinsPrimitive == 'L' or dubinsPrimitive == 'R':
        numSteps = 2000
    else:
        print('unknown dubins primitive supplied: ' + dubinsPrimitive)
        exit(-1)

    return numSteps


def get_control_variable(dubinsPrimitive):
    controlVariable = None
    if dubinsPrimitive == 'S':
        controlVariable = 0.0
    elif dubinsPrimitive == 'L':
        controlVariable = 1.0
    elif dubinsPrimitive == 'R':
        controlVariable = -1.0
    else:
        print('unknown dubins primitive supplied: ' + dubinsPrimitive)
        exit(-1)

    return controlVariable

class DubinsCar:

    def __init__(self, velocity, inputRange, dt=1.0/10000.0):
        self.x = 0.0 
        self.y = 0.0 
        self.theta = 0.0 
        self.velocity = velocity
        self.dt = dt
        self.umin = inputRange[0]
        self.umax = inputRange[1]

    # update car positional state info
    def step(self, u):
        u = clip(u, self.umin, self.umax)
        self.x += self.velocity * math.cos(self.theta) * self.dt
        self.y += self.velocity * math.sin(self.theta) * self.dt
        self.theta += u * self.dt

        return self.get_car_position()

    # current positional state info
    def get_car_position(self):
        return self.x, self.y, self.theta
    
def dubins_car_simulation(testCase = "S"):

    # path state history
    path = {'x': [], 'y': [], 'theta': []}

    # big U is the interval on which little our control variable(angular velocity), can be chosen from
    # see equation 15.43 for details
    maxSteeringAngle = math.pi / 2.0
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]

    # instantiate car model
    dubinsCarModel = DubinsCar(velocity = 0.5, inputRange = U, dt=0.001)

    # iterate through each primitive in word
    for dubinsPrimitive in testCase:

        # set control variable based on right, straight or left direction
        controlVariable = get_control_variable(dubinsPrimitive) 
        numSteps = get_num_steps(dubinsPrimitive)

        # simulate car path
        for i in range(numSteps):
            x, y, theta = dubinsCarModel.step(controlVariable)
            path['x'].append(x)
            path['y'].append(y)
            path['theta'].append(theta)

    # plot path
    plt.plot(path['x'], path['y'], 'o', color='black')
    # car origin
    plt.plot(0.0, 0.0, 'x', color='red', markersize=25)
    plt.title(testCase)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    if len(testCase) == 1:
        pass
    else:
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
    plt.show()

    # save path to mat file
    savemat('./saved-paths/dubins_path_{}.mat'.format(testCase), path)

if __name__ == "__main__":

    dubins_car_simulation('S')
    dubins_car_simulation('R')
    dubins_car_simulation('L')
    dubins_car_simulation('LRL')
    dubins_car_simulation('RLR')
    dubins_car_simulation('LSL')
    dubins_car_simulation('LSR')
    dubins_car_simulation('RSL')
    dubins_car_simulation('RSR')
