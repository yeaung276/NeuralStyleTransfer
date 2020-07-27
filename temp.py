import imageio
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nst_utils import generate_noise_image
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
content_image = imageio.imread('images/louvre_small.jpg')
h = int(generate_noise_image(content_image))
print(np.max(h))
plt.imshow(h[0,:,:,:])
plt.show()