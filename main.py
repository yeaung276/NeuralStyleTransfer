# Author : YeAung
# Email : yeyintaung.ya276@gmail.com
# fbId : https://www.facebook.com/profile.php?id=100009455860398
# download pritrain VGG net here : https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat


import imageio
import scipy.io
import numpy
from PIL import Image
from NST import NST

content_image = imageio.imread('images/louvre_small.jpg')
content_image = NST.reshape_image(content_image) 

style_image = imageio.imread('images/monet.jpg')
style_image = NST.reshape_image(style_image)

config = {
    'img_height' : 300,
    'img_width' : 400,
    'color_channel' : 3,
    'model' : 'pretrained-model/imagenet-vgg-verydeep-19.mat',
    'learning_rate' : 2.0,
    'image_name' : 'pic2',
    'output_dir' : 'output/',
    'style' : [
        ('conv1_1',0.2),
        ('conv2_1',0.2),
        ('conv3_1',0.2),
        ('conv4_1',0.2),
        ('conv5_1',0.2)
        ],
    'content' : [
        ('conv4_2',1)
        ]
}

NST.configure(config)
NST.initialize()

mat = NST.Generate(content_image,style_image,no_iter=200)

scipy.io.savemat('image.mat',mat)
