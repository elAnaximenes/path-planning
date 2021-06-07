import sys
import math
import matplotlib as mpl
import numpy as np
from scipy.optimize import fsolve


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
        car = DubinsOptimalPlanner.dubinsCar
        r = car.umin
        x0, y0 = tuple(DubinsOptimalPlanner.origin[:2])
        xf, yf = tuple(DubinsOptimalPlanner.target[:2])
        firstEqn = (x0*math.cos(alpha)) - (y0*math.sin(alpha)) + (r*math.sin(alpha)) + distance
        secondEqn = (x0*math.sin(alpha)) + (y0*math.cos(alpha)) + (r*(1-math.cos(alpha)))

        return firstEqn, secondEqn

    def calculate_dubins_parameters(self, word):
        if word == "LS":
            alpha, distance = fsolve(_left_straight, (1.0,1.0))



if __name__ == '__main__':
    dubinsCar = 
    



