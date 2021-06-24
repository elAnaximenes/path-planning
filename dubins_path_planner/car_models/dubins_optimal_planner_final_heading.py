import math
from math import sin, cos
import numpy as np
from .dubins_model import DubinsCar

class DubinsOptimalPlannerFinalHeading:

    def __init__(self, dubinsCar, startPosition, target):

        self.dubinsCar = dubinsCar
        self.minTurningRadius = dubinsCar.velocity / dubinsCar.umax
        self.startPosition = startPosition 
        self.target = target
        self._center_car_at_origin()
        self.d = self._calculate_euclidean_distance(startPosition, target)
        self.angularDistanceTraveled = 0.0
        self.linearDistanceTraveled = 0.0
        self.psi = None
        self.alpha = None
        self.beta = None
        self._calculate_alpha_and_beta()

    def _center_car_at_origin(self):

        deltaX = self.target[0] - self.startPosition[0]
        deltaY = self.target[1] - self.startPosition[1]
        theta = self.startPosition[2] 
        phi = self.target[2]

        targetXRelativeToStart = (deltaX * cos(theta)) + (deltaY * sin(theta))
        targetYRelativeToStart = (-1.0 * deltaX * sin(theta)) + (deltaY * cos(theta))
        if phi > theta:
            targetHeadingRelativeToStart = phi - theta
        else:
            targetHeadingRelativeToStart = (2.0 * math.pi) - theta + phi

        self.startPosition = np.array([0.0, 0.0, 0.0])

        self.target = np.array([targetXRelativeToStart, targetYRelativeToStart, targetHeadingRelativeToStart])

    def _calculate_euclidean_distance(self, start, end):

        return abs(np.linalg.norm(start[:2] - end[:2]))

    def _calculate_alpha_and_beta(self):

        xGoal = self.target[0]
        yGoal = self.target[1]
        self.psi = math.acos(xGoal / self.d)

        if yGoal < 0:
            self.psi = (2.0*math.pi) - self.psi

        theta = self.startPosition[-1]
        if self.psi > theta:
            self.alpha = self.psi - theta
        else:
            self.alpha = (2.0 * math.pi) - theta + self.psi

        phi = self.target[-1]
        if self.psi > phi:
            self.beta = self.psi - phi 
        else:
            self.beta = (2.0 * math.pi) - phi + self.psi

    def _calculate_LSL_params(self):

        t = (-1.0 * self.alpha) + (math.atan2((cos(self.beta) - cos(self.alpha)), (d + sin(self.alpha) - sin(self.beta))) % (2.0 * math.pi))
        p = math.sqrt(2.0 + (self.d**2) - (2.0 * cos(self.alpha - self.beta)) + (2.0 * self.d * (sin(self.alpha - self.beta))))
        q = self.beta - (math.atan2((cos(self.beta) - cos(self.alpha)), (d + sin(self.alpha) - sin(self.beta))) % (2.0 * math.pi))

        return t, p, q

    def _calculate_RSR_params(self):

        t = self.alpha - (math.atan2((cos(self.alpha) - cos(self.beta)), (self.d - sin(self.alpha) + sin(self.beta))) % (2.0 * math.pi))
        p = math.sqrt(2.0 + (self.d**2) - (2.0 * cos(self.alpha - self.beta)) + (2.0 * self.d * (sin(self.beta - self.alpha))))
        t = (-1.0 * self.beta % (2.0 * math.pi)) + (math.atan2((cos(self.alpha) - cos(self.beta)), (self.d - sin(self.alpha) + sin(self.beta))) % (2.0 * math.pi))

        return t, p, q

    def _calculate_RSL_params(self):

        p = math.sqrt( (-1.0 * self.alpha) + (self.d ** 2) + (2.0 * cos(self.alpha - self.beta)) + (2.0 * self.d * (sin(self.alpha) + sin(self.beta))))
        t = ((-1.0 * self.alpha) + math.atan2((-1.0 * (cos(self.alpha) + cos(self.beta))), (self.d + sin(self.alpha) + sin(self.beta))) - math.atan2(-2.0, p)) % (2.0 * math.pi)
        q = (-1.0 * (self.beta % (2.0 * math.pi))) + math.atan2((-1.0 * (cos(self.alpha) + cos(self.beta))), (self.d + sin(self.alpha) + sin(self.beta))) - (math.atan2(-2.0, p) % (2.0 * math.pi))

        return t, p, q

    # plan path and steer car to target
    def run(self):

        
        pass

