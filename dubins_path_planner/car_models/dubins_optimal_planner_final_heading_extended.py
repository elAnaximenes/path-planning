import math
from math import sin, cos
import numpy as np
from .dubins_model import DubinsCar
import copy

class DubinsOptimalPlannerFinalHeading:

    def __init__(self, dubinsCar, startPosition, target):

        self.dubinsCar = dubinsCar
        self.minTurningRadius = dubinsCar.velocity / dubinsCar.umax
        self.startPosition = startPosition 
        self.target = target
        self._center_car_at_origin()
        self.euclideanDist = self._calculate_euclidean_distance(startPosition, target) 
        self.d =  self.euclideanDist / self.minTurningRadius

        self.firstCurveDistanceTraveled = 0.0
        self.linearDistanceTraveled = 0.0
        self.secondCurveDistanceTraveled = 0.0
        self.psi = None
        self.alpha = None
        self.beta = None
        self._calculate_alpha_and_beta()
        self.acceptableError = 0.01

    def _center_car_at_origin(self):

        deltaX = self.target[0] - self.startPosition[0]
        deltaY = self.target[1] - self.startPosition[1]
        theta = self.startPosition[2] 
        phi = self.target[2]

        targetXRelativeToStart = (deltaX * cos(theta)) + (deltaY * sin(theta))
        targetYRelativeToStart = (-1.0 * deltaX * sin(theta)) + (deltaY * cos(theta))
        if phi >= theta:
            targetHeadingRelativeToStart = phi - theta
        else:
            targetHeadingRelativeToStart = (2.0 * math.pi) - theta + phi

        self.startPosition = np.array([0.0, 0.0, 0.0])

        self.target = np.array([targetXRelativeToStart, targetYRelativeToStart, targetHeadingRelativeToStart])

    def _calculate_euclidean_distance(self, start, end):

        return abs(np.linalg.norm(start[:2] - end[:2]))

    def _mod_2_pi(self, theta):

        return theta - ((2.0 * math.pi) * math.floor(theta / (2.0 * math.pi)))

    def _calculate_alpha_and_beta(self):

        xGoal = self.target[0]
        yGoal = self.target[1]
        phi = self.target[2]
        theta = self.startPosition[2]

        if self.d > 0:
            self.psi = self._mod_2_pi(math.atan2(yGoal, xGoal))

        self.alpha = self._mod_2_pi(theta - self.psi)
        self.beta = self._mod_2_pi(phi - self.psi)

    def _calculate_LSL_params(self):

        t = self._mod_2_pi((-1.0 * self.alpha) + (math.atan2((cos(self.beta) - cos(self.alpha)), (self.d + sin(self.alpha) - sin(self.beta))))) 
        pSquared = (2.0 + (self.d*self.d) - (2.0 * cos(self.alpha - self.beta)) + (2.0 * self.d * (sin(self.alpha) - sin(self.beta))))
        if pSquared < 0:
            return None, None, None
        p = math.sqrt(pSquared)

        q = self._mod_2_pi(self.beta - (math.atan2((cos(self.beta) - cos(self.alpha)), (self.d + sin(self.alpha) - sin(self.beta)))))

        return t, p, q

    def _calculate_RSR_params(self):

        t = self.alpha - (math.atan2((cos(self.alpha) - cos(self.beta)), (self.d - sin(self.alpha) + sin(self.beta))) % (2.0 * math.pi))
        pSquared = (2.0 + (self.d**2) - (2.0 * cos(self.alpha - self.beta)) + (2.0 * self.d * (sin(self.beta) - sin(self.alpha))))
        if pSquared < 0:
            return None, None, None
        p = math.sqrt(pSquared)
        q = (-1.0 * self.beta % (2.0 * math.pi)) + (math.atan2((cos(self.alpha) - cos(self.beta)), (self.d - sin(self.alpha) + sin(self.beta))) % (2.0 * math.pi))

        return t, p, q

    def _calculate_LSR_params(self):

        pSquared = (self.d*self.d) - 2.0 + (2.0 * cos(self.alpha - self.beta)) + (2.0 * self.d * (sin(self.alpha) + sin(self.beta)))
        if pSquared < 0:
            return None, None, None
        p = math.sqrt(pSquared)
            
        t = self._mod_2_pi((-1.0 * self.alpha) + math.atan2((-1.0 * (cos(self.alpha) + cos(self.beta))), (self.d + sin(self.alpha) + sin(self.beta))) - math.atan2(-2.0, p))
        q = self._mod_2_pi(((-1.0 * self.beta) % (2.0 * math.pi)) + math.atan((-1.0 * (cos(self.alpha) + cos(self.beta)))/ (self.d + sin(self.alpha) + sin(self.beta))) - math.atan(-2.0/ p))

        return t, p, q

    def _calculate_RSL_params(self):

        pSquared = (self.d*self.d) - 2.0 + (2.0 * cos(self.alpha - self.beta)) - (2.0 * self.d * (sin(self.alpha) + sin(self.beta)))
        if pSquared < 0:
            return None, None, None
        p = math.sqrt(pSquared)

        t = self._mod_2_pi(self.alpha - math.atan((cos(self.alpha) + cos(self.beta))/ (self.d - sin(self.alpha) - sin(self.beta))) + math.atan(2.0/ p))
        q = self._mod_2_pi(self.beta - math.atan((cos(self.alpha) + cos(self.beta))/ (self.d - sin(self.alpha) - sin(self.beta))) + math.atan(2.0/ p))

        return t, p, q

    def _calculate_RLR_params(self):

        theta = math.atan2((cos(self.alpha) - cos(self.beta)), (self.d - sin(self.alpha) + sin(self.beta)))
        expr = (6.0 - (self.d*self.d) + (2.0 * cos(self.alpha - self.beta) + (2.0 * self.d * (sin(self.alpha) - sin(self.beta))))) / 8.0

        if abs(expr) > 1.0:
            return None, None, None
        
        p1 = self._mod_2_pi(math.acos(expr))
        t1 = self._mod_2_pi(self.alpha - theta + self._mod_2_pi(p1/2.0))
        q1 = self._mod_2_pi(self.alpha - self.beta - t1 + self._mod_2_pi(p1))

        p2 = self._mod_2_pi((2.0 * math.pi) - math.acos(expr))
        t2 = self._mod_2_pi(self.alpha - theta + self._mod_2_pi(p2/2.0))
        q2 = self._mod_2_pi(self.alpha - self.beta - t2 + self._mod_2_pi(p2))

        if p1 + t1 + q2 < p2 + t2 + q2:
            return t1, p1, q1
        else:
            return t2, p2, q2

    def _calculate_LRL_params(self):

        theta = math.atan2(cos(self.alpha) - cos(self.beta), self.d + sin(self.alpha) - sin(self.beta))
        if (self.d + sin(self.alpha) - sin(self.beta)) < 0.0:
            theta += math.pi

        expr = (6.0 - (self.d*self.d) + (2.0 * cos(self.alpha - self.beta) + (2.0 * self.d * (sin(self.alpha) - sin(self.beta))))) / 8.0

        if abs(expr) > 1.0:
            return None, None, None

        p1 = self._mod_2_pi(math.acos(expr))
        t1 = self._mod_2_pi((-1.0 * self.alpha) - theta + (p1/2.0))
        q1 = self._mod_2_pi(self._mod_2_pi(self.beta) - self.alpha - t1 + self._mod_2_pi(p1))

        p2 = self._mod_2_pi((2.0 * math.pi) - math.acos(expr))
        t2 = self._mod_2_pi((-1.0 * self.alpha) - theta + (p2/2.0))
        q2 = self._mod_2_pi(self._mod_2_pi(self.beta) - self.alpha - t2 + self._mod_2_pi(p2))

        if p1 + t1 + q2 < p2 + t2 + q2:
            return t1, p1, q1
        else:
            return t2, p2, q2

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

        angularVelocity = self._get_angular_velocity(word[-1])

        while self.secondCurveDistanceTraveled < q:

            self.secondCurveDistanceTraveled += (self.dubinsCar.velocity * self.dubinsCar.dt)
            state = self.dubinsCar.step(angularVelocity)

            path['x'].append(self.dubinsCar.state['x'])
            path['y'].append(self.dubinsCar.state['y'])
            path['theta'].append(self.dubinsCar.state['theta'])

        return path

    def _calculate_path_params(self, word):

        if word == 'LSL':
            return self._calculate_LSL_params()
        elif word == 'RSR':
            return self._calculate_RSR_params()
        elif word == 'RSL':
            return self._calculate_RSL_params()
        elif word == 'LSR':
            return self._calculate_LSR_params()
        elif word == 'LRL':
            return self._calculate_LRL_params()
        elif word == 'RLR':
            return self._calculate_RLR_params()

    def _get_shortest_path(self):

        options = ['LSL', 'LSR', 'RSR', 'RSL', 'LRL', 'RLR']

        shortestPath = None
        params = None
        bestWord = None
        for word in options:
            t, p, q = self._calculate_path_params(word)

            if t is None or p is None or q is None:
                continue

            length = abs(t) + abs(p) + abs(q)
            if (shortestPath is None or shortestPath > length):

                shortestPath = length
                bestWord = word
                params = abs(t), abs(p), abs(q)

        return bestWord, params

    # plan path and steer car to target
    def run(self):

        """
        if self.d < math.sqrt(4 - math.pow((abs(cos(self.alpha)) + abs(cos(self.beta))),2)) + abs(sin(self.alpha)) + abs(sin(self.beta)):
            print('short path case')
            return None
        """
        
        word, (t,p,q) = self._get_shortest_path()
        path = self._steer_car_to_target(t,p,q,word)
        
        return path
