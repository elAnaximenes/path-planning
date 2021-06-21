import imageio
import os
images = []
filenames = os.listdir("./batches-for-gif/"
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(',/rrt.gif', images)
