import imageio
import scipy.io
from NST import NST

content_image = imageio.imread('images/louvre_small.jpg')
content_image = NST.reshape_image(content_image) 

style_image = imageio.imread('images/monet.jpg')
style_image = NST.reshape_image(style_image)

config = {
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

image = NST.Generate(content_image,style_image,iter=200)
mat = {
    'image' : image
}
scipy.io.savemat('image.mat',mat)
