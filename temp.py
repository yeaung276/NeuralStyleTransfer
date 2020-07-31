import imageio
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nst_utils import generate_noise_image
from NST import resize_image_keep_ratio,resize_image_forced
# import tensorflow.compat.v1 as tf
# from nst_utils import load_vgg_model



# image = imageio.imread('images/louvre_small.jpg')
# print(image.shape)

# mat = scipy.io.loadmat('pretrained-model/imagenet-vgg-verydeep-19.mat')
# print(mat.keys())

# model = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat')
# print(model['input'].assign())
# dd = tf.Variable(np.array([[0,0]]),dtype='float32')
# c1 = tf.constant(np.array([[1],[2]]))
# d1 = tf.matmul(dd,c1)
# print(dd)
# dd.assign(np.array([[1,2]]))
# print(dd)

# mat = scipy.io.loadmat('image.mat')
# plt.imshow(mat['image'][0,:,:,:])
# plt.show()
# i = mpimg.imread('images/content.jpeg')
# plt.imshow(i)
# plt.show

# draw the figure so the animations will work
# fig = plt.gcf()
# fig.show()
# fig.canvas.draw()
# i = 0
# haha = [0]
# haha2 = [100]
# while True:
#     # compute something
#     haha.append(i)
#     haha2.append(100-i)
#     plt.subplot(1,2,1)
#     plt.plot(haha) # plot something

#     plt.subplot(1,2,2)
#     plt.plot(haha2)
    
#     # update canvas immediately
#     #plt.xlim([0, 100])
#     plt.ylim([0, 100])
#     #plt.pause(0.01)  # I ain't needed!!!
#     fig.canvas.draw()
#     i+=1

# config = {
#     'color_channel' : 3
# }

# print('mean' in config)
# content_image = imageio.imread('images/louvre_small.jpg')
# h = generate_noise_image(content_image)
# print(np.max(h))
# plt.imshow(h[0,:,:,:])
# plt.show()
# x = np.linspace(0, 2*np.pi)
# y1 = np.sin(x)
# y2 = 0.01 * np.cos(x)

# fig = plt.gcf()
# fig.show()
# fig.canvas.draw()
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1)
# ax1.set_ylabel('y1')

# ax2 = ax1.twinx()
# ax2.plot(x, y2, 'r-')
# ax2.set_ylabel('y2', color='r')
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')
# fig.canvas.draw()
# while True : 
#     pass

# import PIL
# from PIL import Image

# mywidth = 300

# img = Image.open('images/akh.jpg')
# wpercent = (mywidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
# img.save('resized.jpg')
# resize_image_keep_ratio(400,'images/akh.jpg','images/akh_resize.jpg')
# resize_image_forced('images/akh_resize.jpg','images/monelisa.jpg','images/monelisa_resize.jpg')
