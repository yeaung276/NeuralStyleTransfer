from PIL import Image
from core.NST import NST
from core.preprocessor import Preprocessor

def setup_nst() -> NST:
    NST.initialize('imagenet-vgg-verydeep-19.mat')
    NST.set_cost_weights(alpha=0.2, beta=0.8)

def nst_handler(c_image: Image, s_image: Image) -> Image:
    processed_c = Preprocessor.transform(c_image)
    processed_s = Preprocessor.transform(s_image)
    print(processed_c.shape, processed_s.shape)
    g_img, _ = NST.generate(processed_c, processed_s, no_iter=100)
    print('gen')
    return g_img