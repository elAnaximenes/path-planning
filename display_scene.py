import json
import matplotlib.pyplot as plt
import sys
from dubins_path_planner.scene import Scene

if len(sys.argv) < 2:
    print('usage: {} scene_name'.format(sys.argv[0]))

sceneName = sys.argv[1]

scene = Scene(sceneName)

plt.title(scene.name)
plt.xlim(scene.dimensions['xmin'], scene.dimensions['xmax'])
plt.ylim(scene.dimensions['ymin'], scene.dimensions['ymax'])

plt.gca().set_aspect('equal')

for obstacle in scene.obstacles:
    obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
    plt.gca().add_patch(obs)

for target in scene.targets:
    tar = plt.Circle((target[0], target[1]), target[2], color='blue', fill=False)
    plt.gca().add_patch(tar)
    
plt.show()
