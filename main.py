import tensorflow.compat.v1 as tf
import imageio
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from nst_utils import load_vgg_model,CONFIG,reshape_and_normalize_image,generate_noise_image

tf.disable_eager_execution()


model = load_vgg_model(CONFIG.VGG_MODEL)

class NST():
    def __init__(self):
        pass
    
    def _calculateNetOutput(self,content,layerName):
        out = None
        with tf.Session() as sess:
            sess.run(model['input'].assign(content))
            out = sess.run(model[layerName])
            sess.close()
        return out

    def _calculateGramMatrix(self,matrix):
        """
        Input : Matrix of shape (c,h*w)
        Output : Gram Matrix(Cross corolation) of shape(c,c)
        """
        return tf.matmul(matrix,tf.transpose(matrix))

    def calculateContentCost(self,content,layerName):
        """
        This Calculate the Content Cost
        A_C.shape = (m,h,w,c), m=1
        A_G.shape = (m,h,w,c), m=1
        """
        A_C = self._calculateNetOutput(content,'conv3_3')
        A_G = model['conv3_3']
        m,h,w,c = A_G.get_shape().as_list()
        J_content = 1/(4 * h * w * c) * tf.reduce_sum(tf.square(tf.subtract(A_C,A_G)))
        return J_content


    def calculateStyleCost(self,style,layerName):
        """
        This Calculate the Style Cost
        A_S.shape = (m,h,w,c) => reshape to (c,h*w), m=1
        A_G.shape = (m,h,w,c) => reshape to (c,h*w), m=1

        """
        A_S = self._calculateNetOutput(style,'conv3_3')
        A_G = model['conv3_3']
        m,h,w,c = A_G.get_shape().as_list()
        A_S = tf.transpose(tf.reshape(A_S,shape=[-1,c]),perm=[1,0])
        A_G = tf.transpose(tf.reshape(A_G,shape=[-1,c]),perm=[1,0])

        #calculate Gram Matrix of each activation
        G_S = self._calculateGramMatrix(A_S)
        G_G = self._calculateGramMatrix(A_G)

        J_style = (1/(4*h*w*c)**2) * tf.reduce_sum(tf.square(tf.subtract(G_S,G_G)))
        return J_style


    def Generate(self,content,style,alpha=10,beta=40,iter=100):
        J_content = self.calculateContentCost(content,[])
        J_style = self.calculateStyleCost(style,[])

        #compute Total Cost
        J = alpha * J_content + beta * J_style

        #Initialize noisy Generated Image
        image = generate_noise_image(content)

        #Set Optimizer
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(J)

        #initialize plot
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()

        #Start the Session
        with tf.Session() as sess:
            #initialize Variables
            sess.run(tf.global_variables_initializer())
            sess.run(model['input'].assign(image))
            J_show = []
            #run optimization
            for i in range(iter):
                sess.run(train_step)
                generated_image = sess.run(model['input'])

                #print infomation
                if i%10 == 0:
                    J_temp = sess.run(J)
                    J_show.append(J_temp)
                    print('iter : {}, J : {}'.format(i,J_temp))
                    plt.subplot(1,2,1)
                    plt.imshow(generated_image[0,:,:,:])
                    fig.canvas.draw()
                    plt.subplot(1,2,2)
                    plt.plot(J_show)
                    plt.xlim(iter)
                    plt.ylim(10**10)

        return generated_image



#code

night = NST()
content_image = imageio.imread('images/louvre_small.jpg')
content_image = reshape_and_normalize_image(content_image) 
style_image = imageio.imread('images/monet.jpg')
style_image = reshape_and_normalize_image(style_image)
image = night.Generate(content_image,style_image)
imshow(image[0,:,:,:])
mat = {
    'image' : image
}
scipy.io.savemat('image.mat',mat)

