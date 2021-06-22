import sys
import math
import numpy as np
from dubins_path_planner.car_models.dubins_model import DubinsCar
from dubins_path_planner.RRT import DubinsCarRRT, Scene

def check_valid_start(scene, startPosition):

    for obstacle in scene.obstacles:

        if abs(np.linalg.norm(startPosition[:2] - obstacle[:2])) < obstacle[2]:
                return False

    return True

def run_RRT(animate=False, sceneName='test_scene'):

    # load scene information
    scene = Scene(sceneName)
    
    validStartPosition = False

    # set car original position
    while not validStartPosition:

        startPosition = np.zeros(3,)
        startPosition[0] = np.random.uniform(scene.dimensions['xmin'], scene.dimensions['xmin'] + 3.0)
        startPosition[1] = np.random.uniform(scene.dimensions['ymin'], scene.dimensions['ymin'] + 3.0)
        startPosition[2] = np.random.uniform(0.0, (0.5 * math.pi))

        #validStartPosition = check_valid_start(scene, startPosition)
        validStartPosition = True

    scene.carStart = startPosition

    # configure and create dubins car
    velocity = 1.0
    maxSteeringAngle = (math.pi / 4.0) 
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
    timeStep = 0.01
    dubinsCar = DubinsCar(startPosition, velocity, U, dt=timeStep)

    # create simulator
    rrtSimulator = DubinsCarRRT(dubinsCar, scene, animate=animate)

    # run RRT algorithm and get final path from car start to target
    sample = rrtSimulator.simulate()

    return sample 

if __name__ == '__main__':
    
    sceneName= 'cluttered_room'
    animate = False

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg != 'animate':
                sceneName = arg

    if len(sys.argv) > 2:
        for arg in sys.argv[1:]:
            if arg == 'animate':
                animate = True

    sample = run_RRT(animate, sceneName)
