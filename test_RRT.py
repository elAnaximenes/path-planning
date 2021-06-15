import sys
import math
from dubins_path_planner.car_models.dubins_model import DubinsCar
from dubins_path_planner.RRT import DubinsCarRRT, Scene

def test_dubins_car_RRT(animate, sceneName):

    # load scene information
    scene = Scene(sceneName)

    # set car original position
    startPosition = scene.carStart

    # configure and create dubins car
    velocity = 1.0
    maxSteeringAngle = (math.pi / 4.0) 
    U = [-1.0 * math.tan(maxSteeringAngle), math.tan(maxSteeringAngle)]
    timeStep = 0.0001
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
        animate = True

    if len(sys.argv) > 2:
        sceneName = sys.argv[2]

    sample = test_dubins_car_RRT(animate, sceneName)
