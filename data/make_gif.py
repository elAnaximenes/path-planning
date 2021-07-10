import imageio
import os
images = []
filenames = os.listdir("./saved_images/")
writer = imageio.get_writer('paths.mp4', format='FFMPEG', mode='I', fps = 1)
for i in range(len(filenames)):
    writer.append_data(imageio.imread(os.path.abspath('./saved_images/img-{}.png'.format(i))))
