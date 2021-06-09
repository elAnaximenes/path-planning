import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from dubins_model import DubinsCar

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

        deltaX = self.target[0] - self.startPosition[0]
        deltaY = self.target[1] - self.startPosition[1]
        r = self.minTurningRadius

        return deltaX, deltaY, r

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
        carDirectionVector = np.array([math.cos(self.startPosition[2]), math.sin(self.startPosition[2])])
        targetInFrontOfCar = np.dot(carDirectionVector, targetVector) > 0

        return targetInFrontOfCar

    def _target_left_of_car(self):
        deltaX, deltaY, r = self.get_path_parameters()

        # cross product direction vector of car and vector from car to target
        # to determine which side of car target is on
        targetVector = np.array([deltaX, deltaY, 0.0])
        carDirectionVector = np.array([math.cos(self.startPosition[2]), math.sin(self.startPosition[2]), 0.0])
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
            
def plot_path(path, origin, target, acceptableError):

    # draw car path as arrows on plot 
    i = 0
    rho = np.linalg.norm(target[:2] - origin[:2])

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
    #plt.savefig('./optimal-{}-{}.png'.format(target[0], target[1]))
    #plt.show()


def simulate_dubins_optimal_path_planner(startPosition, target, animate=True):

        # configure and create dubins car
        velocity = 1.0
        maxSteeringAngle = (math.pi / 4.0) 
        U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
        dubinsCar = DubinsCar(startPosition, velocity, U)

        # create planner
        planner = DubinsOptimalPlanner(dubinsCar, startPosition, target)
        if planner.minTurningRadius > abs(np.linalg.norm(target[:2] - startPosition[:2])):
            print('target within minimum turning radius')
            return

        # get planner's path
        path = planner.run()

        # graph path
        acceptableError = 0.1
        if animate:
            plot_path(path, startPosition, target, acceptableError)

        # test car made it to goal
        carFinalPosition = np.array([path['x'][-1], path['y'][-1]])
        distanceToGoal = abs(np.linalg.norm(carFinalPosition[:2] - target[:2]))
        assert distanceToGoal < acceptableError, 'Car did not reach goal'

def train(animate=True):

    # set starting position and target
    startPosition = np.array([0.0, 0.0, 0.0])
    target = np.array([2.0, 2.0, 0.0])
    simulate_dubins_optimal_path_planner(startPosition, target)

def test(animate=True):
    for i in range(100):
        # set starting position 
        startPosition = np.array([0.0, 0.0, 0.0])

        # set random target
        target = np.random.uniform(low = -10.0, high = 10.0, size = (3,)) 

        # run simulation
        simulate_dubins_optimal_path_planner(startPosition, target, animate)

    print('passed all tests!')


if __name__ == '__main__':

    testOrTrain = 'test'
    animate = False 

    # unpack command line arguments
    if len(sys.argv) < 2:
        print('Usage: {} train/test [animate]'.format(sys.argv[0]))
        exit(2)
    else:
        testOrTrain = sys.argv[1].lower()

    if len(sys.argv) == 3:
        if sys.argv[2].lower() == 'animate':
            animate = True

    # run_simulation
    if testOrTrain== 'train':
        train(animate)
    else:
        test(animate)

