import sys
import os
import math
import numpy as np
import random
import json
from car_models.dubins_model import DubinsCar
from planning_algorithms.adversarial_optimal_RRT import DubinsCarAdversarialOptimalRRT
from scene import Scene

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from classifiers.lstm import LSTM

def get_model(sceneName):

    model = LSTM()
    weightsFile = 'D:\\path_planning_data\\{}_dataset\\optimal_rrt_lstm_weights\\lstm_final_weights'.format(sceneName)
    model.load_weights(weightsFile)

    return model

def check_valid_start(scene, startPosition):

    for obstacle in scene.obstacles:

        if abs(np.linalg.norm(startPosition[:2] - obstacle[:2])) < obstacle[2]:
                return False
    return True

def run_adversarial_optimal_RRT(animate=False, sceneName='test_scene', model=None):

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

    if model is None:
        # configure classifier
        model = get_model(sceneName)

    # create simulator
    optimalRRTSimulator = DubinsCarAdversarialOptimalRRT(dubinsCar, scene, model, animate=animate)

    # run RRT algorithm and get final path from car start to target
    sample = optimalRRTSimulator.simulate()

    return sample 

if __name__ == '__main__':
    
    sceneName= 'tower_defense'
    animate = False 

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg != 'animate':
                sceneName = arg

    if len(sys.argv) > 2:
        for arg in sys.argv[1:]:
            if arg == 'animate':
                animate = True

    for i in range(5,6):

        #seed = random.randint(1, 10000)
        seed = i
        print('seed:', seed)
        random.seed(seed)
        np.random.seed(seed)

        sample = run_adversarial_optimal_RRT(animate, sceneName)

        if sample is None:
            continue

        with open('../data/paths/adversarial_optimal_rrt_path_{}.json'.format(i), 'w') as f:
            json.dump(sample, f)
        print('finished path {}'.format(i), flush=True)
