### Part of this code is due to the MatConvNet team and is used to load the parameters of the pretrained VGG19 model in the notebook ###

import numpy as np


def generate_noise_image(content_image, noise_ratio=0.6):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    _, width, height, channel = content_image.shape
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20,
                                    (1, width, height, channel)).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)
