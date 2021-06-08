import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from dubins_model import DubinsCar


class DubinsOptimalPlanner:

    def __init__(self, dubinsCar, startPosition, target):

        self.dubinsCar = dubinsCar
        self.angularVelocity = dubinsCar.umax
        self.minTurningRadius = dubinsCar.velocity / self.angularVelocity
        self.startPosition = startPosition 
        self.target = target
        self.angularDistanceTraveled = 0.0
        self.linearDistanceTraveled = 0.0

    def get_path_parameters(self):

        deltaX = self.target[0] - self.startPosition[0]
        deltaY = self.target[1] - self.startPosition[1]
        r = self.minTurningRadius

        return deltaX, deltaY, r

    def _left_straight(self, alpha, distance):
        
        arcLength = self.minTurningRadius * alpha 
        path = {'x': [], 'y': [], 'theta': []}

        # turn car
        while self.angularDistanceTraveled < arcLength:
            self.angularDistanceTraveled += (self.dubinsCar.velocity * self.dubinsCar.dt)
            state = self.dubinsCar.step(self.angularVelocity)
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

    def _right_straight(self):
        
        pass


    def _straight_left(self):

        pass
    
    def _straight_right(self):
        
        pass

    def calculate_dubins_parameters(self, word):

        # distance from car to target, minimum turning radius of car
        deltaX, deltaY, r = self.get_path_parameters()

        # turn first
        if word == 'LS' or word == 'RS':
            alpha = -2.0 * math.atan((deltaX - math.pow(((deltaX * deltaX) + (deltaY * deltaY) - (2 * r * deltaY)), (0.5))) / (deltaY - (2 * r)))
            distance = math.pow((deltaX * deltaX) + (deltaY * deltaY) - (2 * r * deltaY), 0.5)
            alpha %= math.pi
             

        # straight first
        elif word == 'SR':
            pass
        elif word == 'SL':
            pass

        return alpha, distance

    def calculate_word(self):

        # x and y distance from start to finish
        deltaX, deltaY, r = self.get_path_parameters()

        # angle of target relative to car starting position
        targetAngleFromCar = math.atan(deltaY / deltaX)

        # is target left or right of car
        if targetAngleFromCar > self.dubinsCar.state['theta']:
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
        if word == 'LS' or word == 'RS':
            path = self._left_straight(alpha, distance)

        # history of car coordinates and orientations
        return path
            
def plot_path(path, origin, target, acceptableError):

    # draw car path as arrows on plot 
    i = 0
    rho = np.linalg.norm(target[:2] - origin[:2])
    print(rho)
    for x,y,theta in zip(path['x'], path['y'], path['theta']):
        if i % (500 * int(rho)) == 0:
            plt.quiver(x, y, math.cos(theta), math.sin(theta)) 
        i += 1

    # text coordinate label shifts
    xShift = 0.1
    yShift = -0.2

    # origin
    plt.plot(origin[0], origin[1], 'x', color='red', markersize=25)
    originStr = 'x: {:.2f}\ny: {:.2f}'.format(origin[0], origin[1])
    plt.text(origin[0] + xShift, origin[1] + yShift, originStr) 

    # target
    targetArea = plt.Circle((target[0], target[1]), acceptableError, color='blue', fill=False)
    plt.gca().add_patch(targetArea)
    targetStr = 'x: {:.2f}\ny: {:.2f}'.format(target[0], target[1])
    plt.text(target[0] + xShift, target[1] + yShift, targetStr) 

    # display
    plt.title('Car Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':

    startPosition = np.array([0.0, 0.0, 0.0])
    target = np.array([2.0, 2.0, 0.0])

    velocity = 1.0
    maxSteeringAngle = (math.pi / 4.0) 
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
    dubinsCar = DubinsCar(startPosition, velocity, U)

    planner = DubinsOptimalPlanner(dubinsCar, startPosition, target)

    path = planner.run()
    acceptableError = 0.1
    plot_path(path, startPosition, target, acceptableError)

