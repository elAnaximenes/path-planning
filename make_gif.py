import imageio
import os
images = []
filenames = os.listdir("./saved-images/")
writer = imageio.get_writer('rrt.mp4', format='FFMPEG', mode='I', fps = 1)
for filename in filenames:
    writer.append_data(imageio.imread(os.path.abspath(os.path.join('./saved-images',filename))))
