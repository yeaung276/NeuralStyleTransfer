import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from typing import Tuple, List
from PIL import Image
from core.model import load_vgg_model
from core.nst_utils import generate_noise_image
from core.preprocessor import Preprocessor

tf.disable_eager_execution()


class NST:
    # image configuration
    height = 300
    width = 400
    channel = 3

    # loss function configuration
    content_layers = [('conv4_2', 1)]
    style_layers = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)
    ]
    alpha = 10
    beta = 40

    # training hypermeter configuration
    learning_rate = 2.0

    # model
    model = None

    # reportings
    fig = None
    ax1 = None
    ax2 = None

    # hypermeter tuning methods
    @classmethod
    def set_image_shape(cls, shape: Tuple[int, int, int]) -> None:
        cls.width, cls.height, cls.channel = shape

    @classmethod
    def set_style_layers(cls, layers: List[Tuple[str, float]]) -> None:
        """set the layer to calculate style cost from VGG19 net

        Args:
            layers (List[Tuple[str, int]]): List containing tuple whose first element
            is layer name and second element is cost multiplier for that layer.
            All the weight should sum up to 1.

        Returns:
            None
        """
        cls.style_layers = layers

    @classmethod
    def set_content_layers(cls, layers: List[Tuple[str, float]]) -> None:
        """set the layer to calculate content cost from VGG19 net

        Args:
            layers (List[Tuple[str, int]]): List containing tuple whose first element
            is layer name and second element is cost multiplier for that layer.
            All the weight should sum up to 1.

        Returns:
            None
        """
        cls.content_layers = layers

    @classmethod
    def set_cost_weights(cls, alpha: float, beta: float) -> None:
        """set the weight for content lost and style loss in loss calculation

        Args:
            alpha (float): weight multiplier for content loss
            beta (float): weight multiplier for style loss
        """
        cls.alpha = alpha
        cls.beta = beta

    @classmethod
    def set_learning_rate(cls, lr: float) -> None:
        cls.learning_rate = lr

    @classmethod
    def initialize(cls, weight_path: str) -> None:
        """
        This function initialize the model for computing activations
        """
        cls.model = load_vgg_model(
            weight_path, (cls.width, cls.height, cls.channel))

    # core math methods
    @classmethod
    def calculate_net_output(cls, input, output_layer_name):
        """
        input : input image to calculate the activation from VGG net,
        output_layer_name : layer from which the activation are to be extracted

        return : activation output of layerName
        """
        with tf.Session() as sess:
            sess.run(cls.model['input'].assign(input))
            out = sess.run(cls.model[output_layer_name])
        return out

    @classmethod
    def calculate_gram_matrix(cls, matrix):
        """
        Input : Matrix of shape (c,h*w)
        Output : Gram Matrix(Cross corolation) of shape(c,c)
        """
        return tf.matmul(matrix, tf.transpose(matrix))

    @classmethod
    def calculate_layer_content_cost(cls, content_image, layer_name):
        """
        This Calculate the Content Cost
        A_C.shape = (m,h,w,c), m=1
        A_G.shape = (m,h,w,c), m=1
        """
        A_C = cls.calculate_net_output(content_image, layer_name)
        A_G = cls.model[layer_name]
        _, h, w, c = A_G.get_shape().as_list()
        J_content = 1/(4 * h * w * c) * \
            tf.reduce_sum(tf.square(tf.subtract(A_C, A_G)))
        return J_content

    @classmethod
    def calculate_layer_style_cost(cls, style_image, layer_name):
        """
        This Calculate the Style Cost
        A_S.shape = (m,h,w,c) => reshape to (c,h*w), m=1
        A_G.shape = (m,h,w,c) => reshape to (c,h*w), m=1

        """
        A_S = cls.calculate_net_output(style_image, layer_name)
        A_G = cls.model[layer_name]
        _, h, w, c = A_G.get_shape().as_list()
        A_S = tf.transpose(tf.reshape(A_S, shape=[-1, c]), perm=[1, 0])
        A_G = tf.transpose(tf.reshape(A_G, shape=[-1, c]), perm=[1, 0])

        # calculate Gram Matrix of each activation
        G_S = cls.calculate_gram_matrix(A_S)
        G_G = cls.calculate_gram_matrix(A_G)

        J_style = (1/(4*h*w*c)**2) * \
            tf.reduce_sum(tf.square(tf.subtract(G_S, G_G)))
        return J_style

    @classmethod
    def calculate_total_content_cost(cls, content):
        """
        Calculate the total content Cost across the VGG conv_net
        """
        J_content = 0
        for layer, coeff in cls.content_layers:
            J_content += coeff * \
                cls.calculate_layer_content_cost(content, layer)
        return J_content

    @classmethod
    def calculate_total_style_cost(cls, style):
        """
        Calculate the total style Cost across the VGG conv_net
        """
        J_style = 0
        for layer, coeff in cls.style_layers:
            J_style += coeff * cls.calculate_layer_style_cost(style, layer)
        return J_style

    # reporting graphs and training data methods

    @classmethod
    def create_fig(cls) -> None:
        cls.fig = plt.gcf()
        cls.fig.set_size_inches(10, 5)
        cls.ax1 = cls.fig.add_subplot(1, 2, 1)
        cls.ax2 = cls.fig.add_subplot(1, 2, 2)

    @classmethod
    def update_fig(cls, generated_image, costs) -> None:
        if (cls.fig == None):
            print('Figure is not initialized.')
            return
        # draw image
        cls.ax1.imshow(Preprocessor.post_process(generated_image))
        cls.ax1.axis('off')
        cls.ax1.set_title('Output image')
        # draw cost
        cls.ax2.plot(costs.get('J_show', []), 'r', label='total cost')
        cls.ax2.plot(costs.get('J_C_show', []), 'g', label="content cost")
        cls.ax2.plot(costs.get('J_S_show', []), 'y', label="style cost")
        cls.ax2.set_ylabel('Cost')
        cls.ax2.set_xlabel('Iteration')
        cls.ax2.legend()
        cls.ax2.set_title('Cost graph')
        cls.fig.canvas.draw()
        plt.show()

    @classmethod
    def generate(cls, content: Image.Image, style: Image.Image, no_iter=100, display=False):
        """
        call signature : Generate(content,style,iter=100)
        input --  content : content image,
                  style   : style image,
                  iter    : number of iteration
        return --- 
                  image : np array of generated image of shape(1,h,w,c)
        """
        if (cls.model == None):
            print('NST is not initialized.')
            return

        J_content = cls.calculate_total_content_cost(content)
        J_style = cls.calculate_total_style_cost(style)

        # compute Total Cost
        J = cls.alpha * J_content + cls.beta * J_style

        # Initialize noisy Generated Image
        image = generate_noise_image(content)

        # Set Optimizer
        optimizer = tf.train.AdamOptimizer(cls.learning_rate)
        train_step = optimizer.minimize(J)

        # initialize plot
        if (display):
            cls.create_fig()
        J_show = []
        J_C_show = []
        J_S_show = []

        # Start the Session
        with tf.Session() as sess:
            # initialize Variables
            sess.run(tf.global_variables_initializer())
            sess.run(cls.model['input'].assign(image))

            # run optimization
            for i in range(no_iter):
                sess.run(train_step)
                generated_image = sess.run(cls.model['input'])

                j, j_c, j_s = sess.run([J, J_content, J_style])
                J_show.append(j)
                J_C_show.append(j_c)
                J_S_show.append(j_s)

                # print infomation
                if i % 10 == 0:
                    print(
                        f"iter : {i+1}, J : {'%.2f' % j}, J_C: {'%.2f' % j_c}, J_S: {'%.2f' % j_s}")

            cost_matrix = {
                "J_show": J_show,
                "J_C_show": J_C_show,
                "J_S_show": J_S_show
            }

            if (display):
                cls.update_fig(
                    generated_image,
                    cost_matrix
                )

        return generated_image, cost_matrix
