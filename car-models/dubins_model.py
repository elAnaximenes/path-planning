# See planning.cs.uiuc.edu chapter 15.3.1(Dubins Curves) for reference material
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# helper function keeps control variable in allowable range
def clip(value, maximum, minimum):

    if value > maximum:
        value = maximum
    elif value < minimum:
        value = minimum
    return value

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

    # current positional state info
    def get_car_position(self):
        return self.x, self.y, self.theta
    
def dubins_car_simulation():

    # big U is the interval on which little our control variable(angular velocity), can be chosen from
    # see equation 15.43 for details
    maxSteeringAngle = math.pi / 2.0
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]

    # instantiate car model
    dubinsCarModel = DubinsCar(velocity = 1.0, inputRange = U)
    controlVariable = 0.0 

    # path state history
    path = {'x': [], 'y': []}

    # simulate car path
    for i in range(100):
        dubinsCarModel.step(controlVariable)
        x, y, theta = dubinsCarModel.get_car_position()
        path['x'].append(x)
        path['y'].append(y)

    # plot path
    plt.plot(path['x'], path['y'], 'o', color = 'black')
    plt.show()

    # save path to mat file
    savemat('dubins_path.mat', path)

if __name__ == "__main__":

    dubins_car_simulation()
