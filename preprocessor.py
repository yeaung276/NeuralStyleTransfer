from typing import Tuple
from PIL import Image
import numpy as np

class Preprocessor:
    WIDTH = 400
    HEIGHT = 300
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    
    @classmethod
    def set_output_shape(cls, shape: Tuple[int, int]) -> None:
        cls.WIDTH, cls.HEIGHT = shape

    @classmethod
    def resize_image(cls, img: Image.Image) -> Image.Image:
        width, height = img.size
        w_percent = cls.WIDTH/width
        new_height = int(height * w_percent)
        return img.resize((cls.WIDTH,new_height), Image.ANTIALIAS)
    
    @classmethod
    def central_crop_image(cls, img: Image.Image) -> Image.Image:
        width, height = img.size
        upper_bound = int(cls.HEIGHT/2)
        lower_bound = cls.HEIGHT - int(cls.HEIGHT/2)
        top = int(height/2) - upper_bound
        bottom = int(height/2) + lower_bound
        return img.crop((0, top, width, bottom))
    
    @classmethod
    def normalize_image(cls, img: Image.Image) -> np.ndarray:
        # Reshape image to mach expected input of VGG16
        img = np.asarray(img)
        img = np.reshape(img, ((1,) + img.shape))
        
        # Substract the mean to match the expected input of VGG16
        img = img - cls.MEANS
        
        return img
    
    @classmethod
    def transform(cls, img_path: str) -> np.ndarray:
        img = Image.open(img_path)
        img = cls.resize_image(img)
        img = cls.central_crop_image(img)
        img = cls.normalize_image(img)
        return img
    
    @classmethod
    def post_process(cls, img) -> Image.Image:
        # Un-normalize the image so that it looks good
        img = img + cls.MEANS
        
        # Clip and Save the image
        img = np.clip(img[0], 0, 255).astype('uint8')
        return Image.fromarray(img)
