import tensorflow.compat.v1 as tf
import imageio
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from nst_utils import load_vgg_model,CONFIG,reshape_and_normalize_image,generate_noise_image,save_image

tf.disable_eager_execution()

model = None

class NST(ABC):
    learning_rate = 2.0
    content = None
    style = None
    name = 'generatedImage.jpg'
    
    @classmethod
    def configure(cls,config):
        """
        config is dictionary containing these fields, 
        If the dictionary didn't specify the field, it will be assign
        default value
        keys ---
        alpha : learning rate of the optimization
        content : array of (layer Name, coefficient) pairs to compute the activation of content image and content cost
        style : array of (layer Name, coefficient) pairs to computte the activation of style image and style cost
        model : path of model .mat file
        color_channel : color channel of content image and style image
        img_height : image height of content image and style image
        img_width : image width of content image and style image
        means : image means
        noise_ratio : noise ratio which will be used in initial noisy image generation
        output_dir : directory for the output image to be saved
        """

        cls.learning_rate = config.get('learning_rate',2.0)
        cls.content = config.get('content',[('conv4_2',1)])
        cls.style = config.get('style',[('conv4_2',1)])
        cls.name = config.get('image_name','generated')
        CONFIG.VGG_MODEL = config.get('model','pretrained-model/imagenet-vgg-verydeep-19.mat')
        CONFIG.COLOR_CHANNELS = config.get('color_channel',3)
        CONFIG.IMAGE_HEIGHT = config.get('img_height',300)
        CONFIG.IMAGE_WIDTH = config.get('img_width',400)
        CONFIG.OUTPUT_DIR = config.get('output_dir','output/')
        if('means' in config):
            CONFIG.MEANS = config['means']
        if('noise_ratio' in config):
            CONFIG.NOISE_RATIO = config['noise_ratio']

    @classmethod
    def initialize(cls):
        """
        This function initialize the model for computing activations
        """
        global model 
        model = load_vgg_model(CONFIG.VGG_MODEL)

    @staticmethod
    def reshape_image(image):
        return reshape_and_normalize_image(image)
        
    @classmethod
    def _calculateNetOutput(cls,inputArg,layerName):
        """
        inputArg : input image to calculate the activation from VGG net,
        layerName : layer from which the activation are to be extracted

        return : activation output of layerName
        """
        out = None
        with tf.Session() as sess:
            sess.run(model['input'].assign(inputArg))
            out = sess.run(model[layerName])
            sess.close()
        return out

    @classmethod
    def _calculateGramMatrix(cls,matrix):
        """
        Input : Matrix of shape (c,h*w)
        Output : Gram Matrix(Cross corolation) of shape(c,c)
        """
        return tf.matmul(matrix,tf.transpose(matrix))

    @classmethod
    def _calculateLayerContentCost(cls,content,layerName):
        """
        This Calculate the Content Cost
        A_C.shape = (m,h,w,c), m=1
        A_G.shape = (m,h,w,c), m=1
        """
        A_C = cls._calculateNetOutput(content,layerName)
        A_G = model[layerName]
        m,h,w,c = A_G.get_shape().as_list()
        J_content = 1/(4 * h * w * c) * tf.reduce_sum(tf.square(tf.subtract(A_C,A_G)))
        return J_content

    @classmethod
    def _calculateLayerStyleCost(cls,style,layerName):
        """
        This Calculate the Style Cost
        A_S.shape = (m,h,w,c) => reshape to (c,h*w), m=1
        A_G.shape = (m,h,w,c) => reshape to (c,h*w), m=1

        """
        A_S = cls._calculateNetOutput(style,layerName)
        A_G = model[layerName]
        m,h,w,c = A_G.get_shape().as_list()
        A_S = tf.transpose(tf.reshape(A_S,shape=[-1,c]),perm=[1,0])
        A_G = tf.transpose(tf.reshape(A_G,shape=[-1,c]),perm=[1,0])

        #calculate Gram Matrix of each activation
        G_S = cls._calculateGramMatrix(A_S)
        G_G = cls._calculateGramMatrix(A_G)

        J_style = (1/(4*h*w*c)**2) * tf.reduce_sum(tf.square(tf.subtract(G_S,G_G)))
        return J_style

    @classmethod
    def calculateTotalContentCost(cls,content,configure):
        """
        Calculate the total content Cost across the VGG conv_net
        """
        J_content = 0
        for layer,coeff in configure:
            J_content += coeff * cls._calculateLayerContentCost(content,layer)
        return J_content

    @classmethod
    def calculateTotalStyleCost(cls,style,configure):
        """
        Calculate the total style Cost across the VGG conv_net
        """
        J_style = 0
        for layer,coeff in configure:
            J_style += coeff * cls._calculateLayerStyleCost(style,layer)
        return J_style

    @staticmethod
    def _createFig():
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        return fig,ax1,ax2

    @staticmethod
    def _updateFig(fig,ax1,ax2,arg1,arg2,x_lim):
        #draw image
        ax1.imshow(arg1[0,:,:,:])
        fig.canvas.draw()
        #draw cost
        J_show,J_C_show,J_S_show = arg2
        ax2.plot(J_show,'r')
        ax2.set_ylabel('total cost')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(J_C_show,'g')
        ax2_twin.plot(J_S_show, 'y')
        ax2_twin.set_ylabel('content and style cost',color='r')
        ax2.set_xlim(int(x_lim/10)+1)
        fig.canvas.draw()

    @classmethod
    def Generate(cls,content,style,alpha=10,beta=40,no_iter=100,display=False):
        """
        call signature : Generate(content,style,alpha=10,beta=40,iter=100)
        input --  content : content image,
                  style   : style image,
                  alpha   : content cost multiplier,
                  beta    : style cost multiplier,
                  iter    : number of iteration
        return --- a dictionary
                  total_cost : array of total cost at eact 10th iteration,
                  content_cost : array of content cost at each 10th iteration,
                  style_cost : array of style cost at each 10th iteration,
                  image : np array of generated image of shape(1,h,w,c)
        side effect ---
                  save the generated image in output dir

        """

        J_content = cls.calculateTotalContentCost(content,cls.content)
        J_style = cls.calculateTotalStyleCost(style,cls.style)

        #compute Total Cost
        J = alpha * J_content + beta * J_style

        #Initialize noisy Generated Image
        image = generate_noise_image(content)

        #Set Optimizer
        optimizer = tf.train.AdamOptimizer(cls.learning_rate)
        train_step = optimizer.minimize(J)

        #initialize plot
        if(display):
            fig,ax1,ax2 = cls._createFig()
        J_show = []
        J_C_show = []
        J_S_show = []

        #Start the Session
        with tf.Session() as sess:
            #initialize Variables
            sess.run(tf.global_variables_initializer())
            sess.run(model['input'].assign(image))

            #run optimization
            for i in range(no_iter):
                sess.run(train_step)
                generated_image = sess.run(model['input'])

                #print infomation
                if i%10 == 0:
                    temp_1,temp_2,temp_3 = sess.run([J,J_content,J_style])
                    J_show.append(temp_1)
                    J_C_show.append(temp_2)
                    J_S_show.append(temp_3)
                    print('iter : {}, J : {}'.format(i,temp_1))
                    if(display):
                        cls._updateFig(fig,ax1,ax2,generated_image,(J_show,J_C_show,J_S_show),no_iter)

        mat = {
            'total_cost' : J_show,
            'content_cost' : J_C_show,
            'style_cost' : J_S_show,
            'image' : generated_image 
        }

        save_image(CONFIG.OUTPUT_DIR + cls.name + '.jpg',generated_image)
        scipy.io.savemat(CONFIG.OUTPUT_DIR + cls.name + '.mat',mat)
        return mat

import PIL
from PIL import Image


def resize_image_keep_ratio(width,imagePath,savePath):
    mywidth = width

    img = Image.open(imagePath)
    wpercent = (mywidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
    print('Image size : {}'.format(img.size))
    img.save(savePath)

def resize_image_forced(imagePath,savePath,*targetImg,size=None):
    resize_size = None
    if(size==None):
        resize_size = Image.open(targetImg).size
    else:
        resize_size = size
    img = Image.open(imagePath)
    img = img.resize(resize_size, PIL.Image.ANTIALIAS)
    print('Image size : {}'.format(img.size))
    img.save(savePath)

