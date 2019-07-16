from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
from Utility import Utility
util = Utility()
from DataProcess import DataStructure

hold_prob = 1
_, _, output_size = util.get_dictionaries()
big_list = list()

class Convolve():  # this uses a keras layer structure but with a custom layer
    def __init__(self, current_list, shape, name):
        self.current_list = current_list
        self.shape = shape
        self.name = name

    def build(self): #input shape is NOT the parameter you feed into convolve's constructor
        self.w_conv_1 = tf.random.truncated_normal(self.shape, stddev =  0.1, name = self.name + "_weight")
        self.current_list.append(self.w_conv_1)

        self.b_conv_1 = tf.zeros(self.shape[3], name=self.name + "_bias")
        self.current_list.append(self.b_conv_1)


    def call(self, input):
        conv_1 = tf.nn.relu(tf.nn.conv2d(input, self.w_conv_1, strides=[1, 1, 1, 1], padding='SAME', name="conv_1"))
        conv_1 = conv_1 + self.b_conv_1
        pooled_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool_1")
        return pooled_1


class Flatten():
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name


    def call(self, input):
        flattened = tf.reshape(input, self.shape, name=self.name + "_flatten")
        return flattened


class Softmax():  # this uses a keras layer structure but with a custom layer

    def call(self, input):
        prediction = tf.nn.softmax(input)
        return prediction


class FC():  # this uses a keras layer structure but with a custom layer
    def __init__(self, current_list, shape, name):
        self.current_list = current_list
        self.shape = shape
        self.name = name

    def build(self):
        self.w_fc_1 = tf.random.truncated_normal(self.shape, stddev=0.1, name=self.name + "_weight")
        self.current_list.append(self.w_fc_1)
        self.b_fc_1 = tf.zeros(self.shape[1], name=self.name + "_bias")
        self.current_list.append(self.b_fc_1)


    def call(self, input):
        fc_1 = tf.matmul(input, self.w_fc_1) + self.b_fc_1
        #fc_1 = tf.nn.dropout(fc_1, rate=1 - hold_prob) removing this for diagnostic purposes
        return fc_1

class Model():
    def __init__(self):
        self.cnn_1 = Convolve(big_list, [8, 8, 1, 32], "Layer_1_CNN")
        self.cnn_2 = Convolve(big_list, [8, 8, 32, 64], "Layer_2_CNN")
        self.cnn_3 = Convolve(big_list, [8, 8, 64, 128], "Layer_3_CNN")

        self.flat = Flatten([-1, 12*12*128], "Fully_Connected")
        self.fc_1 = FC(big_list, [12 * 12 * 128, 2240], "Layer_1_FC")
        self.fc_2 = FC(big_list, [2240, 560], "Layer_1_FC")
        self.fc_3 = FC(big_list, [560, output_size], "Layer_1_FC")
        self.softmax = Softmax()
    def build_model(self):
        self.cnn_1.build()
        self.cnn_2.build()
        self.cnn_3.build()
        self.fc_1.build()
        self.fc_2.build()
        self.fc_3.build()

    @tf.function
    def call(self, input):
        x = self.cnn_1.call(input)
        x = self.cnn_2.call(x)
        x = self.cnn_3.call(x)
        x = self.flat.call(x)
        x = self.fc_1.call(x)
        x = self.fc_2.call(x)
        x = self.fc_3.call(x)
        output = self.softmax.call(x)
        return output

def Big_Train():
    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    model = Model()
    model.build_model()

    for i in range(501):
        data, label = datafeeder.nextBatchTrain_dom(1)
        output = model.call(data)
        print(output)
        input()
        with tf.GradientTape() as tape:
            tape.watch(big_list[1])
            loss = loss_function(output, label)
        gradients = tape.gradient(loss, big_list)
        print(gradients)
        raise Exception

def Conf_mat():
    pass


def main():
    print("---the model is starting-----")
    query = input("What mode do you want? Train (t) or Confusion Matrix (m)?\n")
    if query == "t":
        Big_Train()
    if query == "m":
        Conf_mat()


if __name__ == '__main__':
    main()


