import sys
import math
import numpy as np
import random
import json
import argparse
from car_models.dubins_model import DubinsCar
from planning_algorithms.optimal_RRT import DubinsCarOptimalRRT
from scene import Scene
import matplotlib.pyplot as plt

def check_valid_start(scene, startPosition):

    for obstacle in scene.obstacles:

        if abs(np.linalg.norm(startPosition[:2] - obstacle[:2])) < obstacle[2]:
                return False
    return True

def run_optimal_RRT(animate=False, sceneName='test_scene', target=None):

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
    optimalRRTSimulator = DubinsCarOptimalRRT(dubinsCar, scene, animate=animate, targetIdx=target)

    # run RRT algorithm and get final path from car start to target
    sample = optimalRRTSimulator.simulate()

    return sample#, x, y

if __name__ == '__main__':
    
    sceneName= 'cluttered_room'
    animate = False 

    parser = argparse.ArgumentParser()

    parser.add_argument('--scene', type=str, default='tower_defense')
    parser.add_argument('--animate', default=False, action='store_true')
    parser.add_argument('--target', type=int, default=None)

    args = parser.parse_args()

    sceneName = args.scene
    animate = args.animate
    target = args.target

    costs = np.zeros(400)
    counts = np.zeros(400)

    """
    for i in range(5):
        #seed = random.randint(1, 10000)
        seed = i
        print('seed:', seed)
        random.seed(seed)
        np.random.seed(seed)

        sample = run_optimal_RRT(animate, sceneName, target)
        with open('../data/paths/optimal_rrt_path_{}.json'.format(i), 'w') as f:
            json.dump(sample, f)

        for time, cost in zip(x, y):

            costs[time] += cost
            counts[time] += 1.0

    plt.plot(range(400), costs/counts, 'b--')
    """
    seed = 4
    print('seed:', seed)
    random.seed(seed)
    np.random.seed(seed)


    sample = run_optimal_RRT(animate, sceneName, target)



    
