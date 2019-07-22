import tensorflow as tf
import numpy as np
from Utility import Utility
import pickle
util = Utility()
from DataProcess import DataStructure

_, _, output_size = util.get_dictionaries()


def unpickle(file):
    with open(file, 'rb') as fo:
        objects = pickle.load(fo, encoding='bytes')
    assert len(objects) > 0, "there are no weights saved"
    return objects

class Convolve():
    def __init__(self, current_list, shape, name):
        self.current_list = current_list
        self.shape = shape
        self.name = name

    def build(self, from_file = False, weights = None): #input shape is NOT the parameter you feed into convolve's constructor
        if not(from_file):
            self.w_conv = tf.Variable(initial_value = tf.random.truncated_normal(self.shape, stddev =  0.1),
                                        name = self.name + "_weight", trainable = True)
            self.current_list.append(self.w_conv)

            self.b_conv = tf.Variable(initial_value = tf.zeros(self.shape[3]),
                                        name=self.name + "_bias", trainable = True)
            self.current_list.append(self.b_conv)
        else:
            print("Loading filters from saved weights")
            assert np.shape(weights[0]) == self.shape, "Shape mis-match in class Convolve"

            self.w_conv = tf.Variable(initial_value=weights[0],
                                        name=self.name + "_weight", trainable=True)
            self.current_list.append(self.w_conv)

            self.b_conv = tf.Variable(initial_value=weights[1],
                                        name=self.name + "_bias", trainable=True)
            self.current_list.append(self.b_conv)


    def call(self, input):
        conv = tf.nn.relu(tf.nn.conv2d(input, self.w_conv, strides=[1, 1, 1, 1], padding='SAME', name="conv"))
        conv = conv + self.b_conv
        return conv

    def l2loss(self):
        l2 = tf.reduce_sum(tf.abs(self.w_conv))
        return l2

class Pool():
    def call(self, input):
        pooled = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool")
        return pooled


class Flatten():
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name

    def call(self, input):
        flattened = tf.reshape(input, self.shape, name=self.name + "_flatten")
        return flattened


class Dropout():
    def __init__(self, name, hold_prob):
        self.name = name
        self.hold_prob = hold_prob

    def call(self, input):
        output = tf.nn.dropout(input, rate=1 - self.hold_prob, name = self.name)
        return output

class Softmax():
    def call(self, input):
        prediction = tf.nn.softmax(input)
        return prediction

class FC():
    def __init__(self, current_list, shape, name):
        self.current_list = current_list
        self.shape = shape
        self.name = name

    def build(self, from_file = False, weights = None): #I am working on this right now
        if not(from_file):
            self.w_fc = tf.Variable(initial_value = tf.random.truncated_normal(self.shape, stddev=0.1),
                                      name=self.name + "_weight", trainable = True)
            self.current_list.append(self.w_fc)
            self.b_fc = tf.Variable(initial_value = tf.zeros(self.shape[1]),
                                      name=self.name + "_bias", trainable = True)
            self.current_list.append(self.b_fc)
        else:
            print("Loading neurons from saved weights")
            assert np.shape(weights[0]) == self.shape, "shape mis-match in FC"
            self.w_fc = tf.Variable(initial_value=weights[0],
                                      name=self.name + "_weight", trainable=True)
            self.current_list.append(self.w_fc)
            self.b_fc = tf.Variable(initial_value=weights[1],
                                      name=self.name + "_bias", trainable=True)
            self.current_list.append(self.b_fc)


    def call(self, input):
        fc = tf.matmul(input, self.w_fc) + self.b_fc
        return fc





