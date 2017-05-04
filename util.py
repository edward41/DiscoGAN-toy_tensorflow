
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os


''' 
Pytorch 

'nn.Linear'
http://pytorch.org/docs/_modules/torch/nn/modules/linear.html#Linear

def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)
'''
def linear(input,output,name='None') :
    import math
    stdv=1./ math.sqrt(input.get_shape().as_list()[1])

    with tf.variable_scope(name) :
        '''
        W=tf.Variable(tf.zeros([input.get_shape().as_list()[1], output],dtype=tf.float32), name = 'w')
        b=tf.Variable(tf.zeros([output],dtype=tf.float32), name = 'b')
        '''

        '''tf.contrib'.layers.xavier_initializer()'''

        W = tf.get_variable("weights", [input.get_shape().as_list()[1],output],
                                  initializer=tf.contrib.layers.xavier_initializer())
        # Create variable named "biases".
        b = tf.get_variable("biases", [output],
                                 initializer=tf.contrib.layers.xavier_initializer())



        return tf.matmul(input,W)+b

def file_clear(dir) :
    try:
        file_list = os.listdir(dir)
        print file_list
        for file in file_list:
            os.remove(os.path.join(dir, file))
    except Exception as e:
        print(e)

def plot(data, color):
    plt.plot(data[:,0], data[:,1], color)

def plot_with_class(data_with_class):
        for key, value in data_with_class.items():
            plot(value, '.')

def generate_data(num_mode, except_num, radius=2,
                  center=(0, 0), sigma=0.1, num_data_per_class=20000):
    total_data = {}

    t = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(t) * radius + center[0]
    y = np.sin(t) * radius + center[1]

    #plt.figure()
    #plt.plot(x, y)

    modes = np.vstack([x, y]).T

    for idx, mode in enumerate(modes[except_num:]):
        x = np.random.normal(mode[0], sigma, num_data_per_class)
        y = np.random.normal(mode[1], sigma, num_data_per_class)
        total_data[idx] = np.vstack([x, y]).T

        #plt.plot(x, y)
    #plt.figure()
    all_points = np.vstack([values for values in total_data.values()])
    data_x, data_y = all_points[:, 0], all_points[:, 1]

    return total_data, all_points