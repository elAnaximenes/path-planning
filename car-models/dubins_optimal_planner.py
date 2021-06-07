import sys
import math
import matplotlib as mpl
import numpy as np
from scipy.optimize import fsolve
from dubins_model import DubinsCar

class DubinsOptimalPlanner:

    def __init__(self, dubinsCar, origin, target):

        self.dubinsCar = dubinsCar
        self.straightDistance = 0.0
        self.turningDistance = 0.0
        self.origin = origin
        self.target = target

    @staticmethod
    def _left_straight(parameters):

        alpha, distance = parameters
        car = self.dubinsCar
        r = car.velocity / car.umin
        x0, y0 = tuple(self.origin[:2])
        xf, yf = tuple(self.target[:2])
        firstEqn = (x0*math.cos(alpha)) - (y0*math.sin(alpha)) + (r*math.sin(alpha)) + distance
        secondEqn = (x0*math.sin(alpha)) + (y0*math.cos(alpha)) + (r*(1-math.cos(alpha)))

        return firstEqn, secondEqn

    @staticmethod
    def _right_straight(parameters, car, origin, target):

        alpha, distance = parameters
        car = self.dubinsCar
        r = car.velocity / car.umin
        x0, y0 = tuple(self.origin[:2])
        xf, yf = tuple(self.target[:2])
        firstEqn = (x0*math.cos(alpha)) - (y0*math.sin(alpha)) + (r*(1-math.cos(alpha))) + distance
        secondEqn = (x0*math.sin(alpha)) + (y0*math.cos(alpha)) + (r*(math.sin(alpha)))

        return firstEqn, secondEqn

    def calculate_dubins_parameters(self, word):

        if word == 'LS':
            alpha, distance = fsolve(self._left_straight, [0.0,0.0])
        if word == 'RS':
            alpha, distance = fsolve(self._right_straight, [0.0,0.0])

        return alpha, distance

if __name__ == '__main__':

    origin = np.array([0.0, 0.0, 0.0])
    target = np.array([10.0, -10.0, 0.0])

    velocity = 1.0
    maxSteeringAngle = math.pi / 2.0
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
    dubinsCar = DubinsCar(origin, velocity, U)

    planner = DubinsOptimalPlanner(dubinsCar, origin, target)

    print(planner.calculate_dubins_parameters('LS'))
    print(planner.calculate_dubins_parameters('RS'))
