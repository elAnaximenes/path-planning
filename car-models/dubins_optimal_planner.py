import sys
import math
import matplotlib as mpl
import numpy as np


class DubinsOptimalPlanner:

    def __init__(self, dubinsCar, origin, target):
        self.dubinsCar = dubinsCar
        self.straightDistance = 0.0
        self.turningDistance = 0.0
        self.origin = origin
        self.target = target

    def 

