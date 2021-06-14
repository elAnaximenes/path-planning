import matplotlib.pyplot as plt
import json
import RRT

def draw_scene(scene, path, sampleNum):

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title('Simple Room Path - {}'.format(sampleNum))
    plt.xlim(scene.dimensions['xmin'] - 1.0, scene.dimensions['xmax'] + 1.0)
    plt.ylim(scene.dimensions['ymin'] - 1.0, scene.dimensions['ymax'] + 1.0)

    ax.set_aspect('equal')
    ax.set_ylabel('Y-distance(M)')
    ax.set_xlabel('X-distance(M)')
    ax.plot(path['x'][0], path['y'][0], 'x')

    for obstacle in scene.obstacles:
        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        ax.add_patch(obs)
        fig.canvas.draw()

    for target in scene.targets:

        tar = plt.Circle((target[0], target[1]), target[2], color='blue', fill=False)
        ax.add_patch(tar)
        fig.canvas.draw()
    
    ax.plot(path['x'], path['y'], color=colorSelection, linestyle='-', markersize=2)
        
    plt.show()
    plt.savefig('./presentations/june_14/paths/{}-{}.png'.format(scene.name.replace(' ', '_'), sampleNum))
    

def load_paths(sceneName, batchNum):
    print('loading paths')
    with open('./paths/{}_batch_{}.json'.format(sceneName, batchNum), 'r') as f:
        path = json.load(f)
    print('paths loaded')
    return path




sceneName = 'simple_room'
scene = RRT.Scene(sceneName)
print('scene loaded')

batchNum = 0
numSamples = 100
paths = load_paths(sceneName, batchNum)
for i in range(100):
    print('drawing path {}'.format(i))
    path = paths[''.format(i)]['path']

    draw_scene(scene, path, i)



