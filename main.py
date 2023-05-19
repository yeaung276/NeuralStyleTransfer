# Author : YeAung
# Email : yeyintaung.ya276@gmail.com
# download pritrain VGG net here : https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat


from core.preprocessor import Preprocessor
from core.NST import NST
import matplotlib

matplotlib.use('TkAgg')

content_image = Preprocessor.transform('images/louvre_small.jpg')

style_image = Preprocessor.transform('images/monet.jpg')

NST.initialize('imagenet-vgg-verydeep-19.mat')

NST.set_cost_weights(alpha=0.2, beta=0.8)

g_img, _ = NST.generate(content_image,style_image,no_iter=100, display=True)

processed_img = Preprocessor.post_process(g_img)
processed_img.save('output.png')
