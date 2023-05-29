from PIL import Image
from core.NST import NST
from core.preprocessor import Preprocessor

##### hyperparameters #####
no_iter = 50
alpha = 0.2
beta = 0.8


def setup_nst() -> NST:
    NST.initialize('imagenet-vgg-verydeep-19.mat')
    NST.set_cost_weights(alpha=alpha, beta=beta)

def nst_handler(c_image: Image, s_image: Image) -> Image:
    processed_c = Preprocessor.transform(c_image)
    processed_s = Preprocessor.transform(s_image)
    g_img, _ = NST.generate(processed_c, processed_s, no_iter=no_iter)
    return Preprocessor.post_process(g_img)