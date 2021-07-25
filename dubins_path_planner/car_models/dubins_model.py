# See planning.cs.uiuc.edu chapter 15.3.1(Dubins Curves) for reference material
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

class DubinsCar:

    def __init__(self, initialState, velocity, inputRange, dt=0.01):

        self.state = {}
        self.set_state(initialState)
        self.velocity = velocity
        self.dt = dt
        self.umin = inputRange[0]
        self.umax = inputRange[1]
        self.minTurningRadius = abs(self.velocity / self.umax)

    def set_state(self, newState):
        self.state['x'] = newState[0]
        self.state['y'] = newState[1]
        self.state['theta'] = newState[2]


    # update car positional state info
    def step(self, u):

        u = self._clip(u, self.umin, self.umax)
        self.state['x'] += self.velocity * math.cos(self.state['theta']) * self.dt
        self.state['y'] += self.velocity * math.sin(self.state['theta']) * self.dt
        self.state['theta'] += u * self.dt
        self.state['theta'] %= (2.0 * math.pi)

        return self.state

    # helper function keeps control variable in allowable range
    def _clip(self, value, minimum, maximum):

        if value > maximum:
            value = maximum
        elif value < minimum:
            value = minimum

        return value

def get_control_input(dubinsPrimitive):

    controlInput = None
    if dubinsPrimitive == 'S':
        controlInput = 0.0
    elif dubinsPrimitive == 'L':
        controlInput = 1.0
    elif dubinsPrimitive == 'R':
        controlInput = -1.0
    else:
        print('unknown dubins primitive supplied: ' + dubinsPrimitive)
        exit(-1)

    return controlInput

def test_dubins_car(dubinsCarModel, testCase):

    # path state history
    path = {'x': [], 'y': [], 'theta': []}

    # iterate through each primitive in word
    for dubinsPrimitive in testCase:

        # set control variable based on right, straight or left direction
        controlInput = get_control_input(dubinsPrimitive) 
        numSteps = 30000

        # simulate car path
        for i in range(numSteps):
            state = dubinsCarModel.step(controlInput)
            path['x'].append(state['x'])
            path['y'].append(state['y'])
            path['theta'].append(state['theta'])

    return path

def plot_path(path, testCase):

    plt.plot(path['x'], path['y'], 'o', color='black') # car path
    plt.plot(0.0, 0.0, 'x', color='red', markersize=25)# car origin
    plt.title(testCase)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis("equal")
    plt.show()

def dubins_car_simulation(testCase = "S"):

    # big U is the interval of possible control variable(angular velocity) values
    # see equation 15.43 for details
    maxSteeringAngle = math.pi / 2.0
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]

    # instantiate car model
    initialState = [0.0, 0.0, 0.0]
    dubinsCarModel = DubinsCar(initialState, velocity = 1.0, inputRange = U, dt=0.0001)

    # test dubins car for this primitive
    path = test_dubins_car(dubinsCarModel, testCase)

    # show path of car
    plot_path(path, testCase)

    # save path to mat file
    # savemat('./saved-paths/dubins_path_{}.mat'.format(testCase), path)

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
