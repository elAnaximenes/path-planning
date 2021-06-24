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
        self.firstCurveDistanceTraveled = 0.0
        self.firstCurveDistanceTraveled = 0.0
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
        q = (-1.0 * self.beta % (2.0 * math.pi)) + (math.atan2((cos(self.alpha) - cos(self.beta)), (self.d - sin(self.alpha) + sin(self.beta))) % (2.0 * math.pi))

        return t, p, q

    def _calculate_RSL_params(self):

        p = math.sqrt( (-1.0 * self.alpha) + (self.d ** 2) + (2.0 * cos(self.alpha - self.beta)) + (2.0 * self.d * (sin(self.alpha) + sin(self.beta))))
        t = ((-1.0 * self.alpha) + math.atan2((-1.0 * (cos(self.alpha) + cos(self.beta))), (self.d + sin(self.alpha) + sin(self.beta))) - math.atan2(-2.0, p)) % (2.0 * math.pi)
        q = (-1.0 * (self.beta % (2.0 * math.pi))) + math.atan2((-1.0 * (cos(self.alpha) + cos(self.beta))), (self.d + sin(self.alpha) + sin(self.beta))) - (math.atan2(-2.0, p) % (2.0 * math.pi))

        return t, p, q

    def _calcuate_LSR_params(self):

        p = math.sqrt((self.d**2) - 2.0 + (2.0 * cos(self.alpha - self.beta)) - (2.0 * self.d * (sin(self.alpha) + sin(self.beta))))
        t = self.alpha - math.atan2((cos(self.alpha) + cos(self.beta)), (self.d - sin(self.alpha) - sin(self.beta))) + (math.atan2(2.0, p) % (2.0 * math.pi))
        q = (self.beta % (2.0 * math.pi)) - math.atan2((cos(self.alpha) + cos(self.beta)), (self.d - sin(self.alpha) - sin(self.beta))) + (math.atan2(2.0, p) % (2.0 * math.pi))

        return t, p, q

    def _get_angular_velocity(self, letter):

        if letter == 'L':
            return self.dubinsCar.umax
        elif letter == 'R':
            return self.dubinsCar.umin
        else:
            print('Unrecognized turning character:', letter)
            exit(-1)

    def _steer_car_to_target(self, t, p, q, word):

        path = {'x': [], 'y': [], 'theta': []}
        angularVelocity = self._get_angular_velocity(word[0])

        while self.firstCurveDistanceTraveled < t:

            self.firstCurveDistanceTraveled += (self.dubinsCar.velocity * self.dubinsCar.dt)
            state = self.dubinsCar.step(angularVelocity)

            path['x'].append(self.dubinsCar.state['x'])
            path['y'].append(self.dubinsCar.state['y'])
            path['theta'].append(self.dubinsCar.state['theta'])

        while self.linearDistanceTraveled < p:

            self.linearDistanceTraveled += (self.dubinsCar.velocity * self.dubinsCar.dt)
            state = self.dubinsCar.step(0.0)

            path['x'].append(self.dubinsCar.state['x'])
            path['y'].append(self.dubinsCar.state['y'])
            path['theta'].append(self.dubinsCar.state['theta'])

        angularVelocity = self.get_angular_velocity(word[-1])

        while self.secondCurveDistanceTraveled < q:

            self.firstCurveDistanceTraveled += (self.dubinsCar.velocity * self.dubinsCar.dt)
            state = self.dubinsCar.step(angularVelocity)

            path['x'].append(self.dubinsCar.state['x'])
            path['y'].append(self.dubinsCar.state['y'])
            path['theta'].append(self.dubinsCar.state['theta'])

        return path

    def _calculate_path_params(self, word):

        if word = 'LSL':
            return self._claculate_LSL_params()
        elif word = 'RSR':
            return self._claculate_RSR_params()
        elif word = 'RSL':
            return self._claculate_RSL_params()
        elif word = 'LSR':
            return self._claculate_LSR_params()

    def _get_quadrant(self, angle):
    
        assert angle >= 0.0, 'Angle cannot be negative'

        if angle < (0.5 * math.pi):
            return 1
        elif angle < (math.pi):
            return 2
        elif angle < (1.5 * math.pi):
            return 3
        else:
            return 4

    def _get_word(self):

        alpha_quadrant = self._get_quadrant(self.alpha)
        beta_quadrant = self._get_quadrant(self.beta)

        return 'LSR'

    # plan path and steer car to target
    def run(self):

        word = self._get_word()
        t, p, q = self._calculate_path_params(word)
        path = self._steer_car_to_target(t,p,q,word)
        
        return path

