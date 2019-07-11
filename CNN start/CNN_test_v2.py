from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
var_list = list()
def make_weights(shape, name):
    weight_name = name + "_Weight"
    with tf.name_scope("Weights"):
        distribution = tf.random.truncated_normal(shape, stddev =  0.1, name = weight_name)
        var = tf.Variable(distribution)
        tf.summary.histogram(weight_name, var)
        var_list.append(var)
        return var

def make_bias(shape, name):
    bias_name = name + "_Bias"
    with tf.name_scope("Biases"):
        distribution = tf.constant(0.1, shape = shape, name = bias_name)
        var = tf.Variable(distribution)
        tf.summary.histogram(bias_name, var)
        var_list.append(var)
        return var

def conv2d(input, filter, name):
    conv_name = name +"_Conv"
    with tf.name_scope("Conv"):
        output = tf.nn.conv2d(input, filter, strides = [1,1,1,1], padding = 'SAME', name = conv_name) #this essentially does the filter operation keeping the dimensions the same
        return output

def max_pool(input, name):
    pool_name = name + "_Pool"
    with tf.name_scope("Pool"):
        output = tf.nn.max_pool(input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = pool_name) #ksize is the window size, and it just pools every 4 grid into 1 (need to work out why later)
        return output

def convolutional_layer(input, shape, name):
    W = make_weights(shape, name)
    b = make_bias([shape[3]], name)
    return tf.nn.relu(conv2d(input, W, name) + b)

def fully_connected(input, end_size, name):
    in_size = int(input.get_shape()[1])
    W = make_weights([in_size, end_size], name)
    b = make_bias([end_size], name)
    return(tf.matmul(input, W) + b)

@tf.function
def model(input, hold_prob):
    with tf.name_scope("Layer_1"):
        conv_1 = convolutional_layer(input, shape = [4,4,3,32], name = "Layer_1") # 4 and 4 is the window, 3 is the color channels and 32 is number of output layers (filters)
        conv_1_pooled = max_pool(conv_1, name = "Layer_1")

    with tf.name_scope("Layer_2"):
        conv_2 = convolutional_layer(conv_1_pooled, shape = [4,4,32,64], name = "Layer_2")
        conv_2_pooled = max_pool(conv_2, name = "Layer_2")

    with tf.name_scope("Layer_3"):
        conv_3 = convolutional_layer(conv_2_pooled, shape=[4, 4, 64, 128], name="Layer_3")
        conv_3_pooled = max_pool(conv_3, name="Layer_3")

    with tf.name_scope("Fully_Connected"):
        flattened = tf.reshape(conv_3_pooled, [-1, 4*4*128], name = "Flatten")
        fc_1 = fully_connected(flattened, 1024, name = "Fully_Connected_Layer_1")
        dropout_1 = tf.nn.dropout(fc_1, rate = 1-hold_prob)

    with tf.name_scope("Output"):
        prediction = fully_connected(dropout_1, 10, name = "raw_pred")
        prediction = tf.nn.softmax(prediction)

    return prediction


def Big_Train():

    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.sparse_categorical_crossentropy()

    display, _ = datafeeder.nextBatchTrain(10)
    tf.compat.v1.summary.image("10 training data examples", display, max_outputs=10)
    for i in range(501):
        data, label = datafeeder.nextBatchTrain(100)
        output = model(data, 0.6)
        with tf.GradientTape() as tape:
            loss = loss_function(y_true = label, y_pred = output)

        grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))
        print(loss)



def main():
    print("---the model is starting-----")
    query = input("What mode do you want? Train (t) or Confusion Matrix (m)?\n")
    if query == "t":
        Big_Train()


if __name__ == '__main__':
    main()

