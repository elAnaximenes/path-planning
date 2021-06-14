import os
import matplotlib.pyplot as plt
import imageio

with imageio.get_writer('optimal_planner.gif', mode='I', fps=1) as writer:
    for i in range(10):
        imageName = 'optimal-{}.png'.format(i)
        image = imageio.imread(imageName)
        writer.append_data(image)
