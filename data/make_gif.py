import imageio
import os
images = []
filenames = os.listdir("./saved_images/")
writer = imageio.get_writer('paths.mp4', format='FFMPEG', mode='I', fps = 1)
for filename in filenames:
    writer.append_data(imageio.imread(os.path.abspath(os.path.join('./saved_images',filename))))
