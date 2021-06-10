import math
import numpy as np
from .dubins_model import DubinsCar

"""""""""""
Usage: dubins_optimal_planner.py animate test
"""""""""""

class DubinsOptimalPlanner:

    def __init__(self, dubinsCar, startPosition, target):

        self.dubinsCar = dubinsCar
        self.minTurningRadius = dubinsCar.velocity / dubinsCar.umax
        self.startPosition = startPosition 
        self.target = target
        self.angularDistanceTraveled = 0.0
        self.linearDistanceTraveled = 0.0

    def get_path_parameters(self):

        r = self.minTurningRadius

        deltaX = self.target[0] - self.startPosition[0]
        deltaY = self.target[1] - self.startPosition[1]
        theta = self.startPosition[2] 

        targetXRelativeToStart = (deltaX * math.cos(theta)) + (deltaY * math.sin(theta))
        targetYRelativeToStart = (-1.0 * deltaX * math.sin(theta)) + (deltaY * math.cos(theta))

        return targetXRelativeToStart, targetYRelativeToStart, r

    def calculate_dubins_parameters(self, word):

        # distance from car to target, minimum turning radius of car
        deltaX, deltaY, r = self.get_path_parameters()
        alpha = 0.0
        distance = 0.0

        deltaX = abs(deltaX)
        deltaY = abs(deltaY)

        # turn first
        if word == 'LS' or word == 'RS':
            if self._target_in_front_of_car():
                alpha = -2.0 * math.atan((deltaX - math.pow(((deltaX * deltaX) + (deltaY * deltaY) - (2 * r * deltaY)), (0.5))) / (deltaY - (2 * r)))
            else:
                alpha = -2.0 * math.atan2((deltaX + math.pow(((deltaX * deltaX) + (deltaY * deltaY) - (2 * r * deltaY)), (0.5))), (deltaY - (2 * r)))
            distance = math.pow((deltaX * deltaX) + (deltaY * deltaY) - (2 * r * deltaY), 0.5)
            
        # drive straight first
        elif word == 'SR':
            pass

        
        return alpha, distance

    def _target_in_front_of_car(self):

        deltaX, deltaY, r = self.get_path_parameters()

        # dot product car direction and distance vector to target to determine
        # if target is initially in front of or behind car
        targetVector = np.array([deltaX, deltaY])
        carDirectionVector = np.array([1.0, 0.0])
        targetInFrontOfCar = np.dot(carDirectionVector, targetVector) > 0

        return targetInFrontOfCar

    def _target_left_of_car(self):
        
        deltaX, deltaY, r = self.get_path_parameters()

        # cross product direction vector of car and vector from car to target
        # to determine which side of car target is on
        targetVector = np.array([deltaX, deltaY, 0.0])
        carDirectionVector = np.array([1.0, 0.0, 0.0])
        targetLeftOfCar = np.cross(carDirectionVector, targetVector)[2] > 0

        return targetLeftOfCar 

    def _turn_straight(self, alpha, distance, angularVelocity):
        
        arcLength = abs(self.minTurningRadius * alpha)
        path = {'x': [], 'y': [], 'theta': []}

        # turn car
        while self.angularDistanceTraveled < arcLength:

            self.angularDistanceTraveled += (self.dubinsCar.velocity * self.dubinsCar.dt)
            state = self.dubinsCar.step(angularVelocity)

            path['x'].append(self.dubinsCar.state['x'])
            path['y'].append(self.dubinsCar.state['y'])
            path['theta'].append(self.dubinsCar.state['theta'])

        # drive car straight to goal
        while self.linearDistanceTraveled < distance:

            self.linearDistanceTraveled += (self.dubinsCar.velocity * self.dubinsCar.dt)
            state = self.dubinsCar.step(0.0)

            path['x'].append(self.dubinsCar.state['x'])
            path['y'].append(self.dubinsCar.state['y'])
            path['theta'].append(self.dubinsCar.state['theta'])

        return path

    def _straight_turn(self):
        
        pass

    def calculate_word(self):

        deltaX, deltaY, r = self.get_path_parameters()

        # is target left or right of car
        if self._target_left_of_car():
            word = 'L'
        else:
            word = 'R'

        rho = np.linalg.norm(self.target[:2] - self.startPosition[:2])

        # is target within the car's maximum turning radius
        if rho > r:
            word += 'S'
        else:
            word = 'S' + word

        return word 

    # plan path and steer car to target
    def run(self):

        # get correct dubins primitive(RS, LS, SR, SL)
        word = self.calculate_word()
        
        # based on primitive, get angle of turning arc and straight distance
        alpha, distance = self.calculate_dubins_parameters(word)

        # steer car to target
        if word == 'LS':
            path = self._turn_straight(alpha, distance, self.dubinsCar.umax)
        elif word == 'RS':
            path = self._turn_straight(alpha, distance, self.dubinsCar.umin)

        # history of car coordinates and orientations
        return path

