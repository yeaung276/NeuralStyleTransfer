# Author : YeAung
# Email : yeyintaung.ya276@gmail.com
# download pritrain VGG net here : https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat


from preprocessor import Preprocessor
from NST import NST

content_image = Preprocessor.transform('images/louvre_small.jpg')

style_image = Preprocessor.transform('images/monet.jpg')

NST.initialize('imagenet-vgg-verydeep-19.mat')

g_img = NST.generate(content_image,style_image,no_iter=200)

processed_img = Preprocessor.post_process(g_img)
processed_img.save('output.png')
