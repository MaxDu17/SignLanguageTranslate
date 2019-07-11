from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
var_list = list()


class CustomLayer(tf.keras.layers.Layer): #this uses a keras layer structure but with a custom layer
    def __init__(self, *args, **kwargs):
        super(CustomLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape=None):
        self.w_conv_1 = self.add_weight(
            shape=[4,4,3,32],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.b_conv_1 = self.add_bias(
            shape=[32],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.w_conv_2 = self.add_weight(
            shape=[4, 4, 32, 64],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.b_conv_2 = self.add_bias(
            shape=[64],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.w_conv_3 = self.add_weight(
            shape=[4, 4, 64, 128],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.b_conv_3 = self.add_bias(
            shape=[128],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.w_fc_1 = self.add_weight(
            shape=[4*4*128, 1024],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.b_fc_1 = self.add_bias(
            shape=[1024],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.w_fc_2  = self.add_weight(
            shape=[1024,10],
            dtype=tf.float32,
            initializer=tf.keras.initializers.ones(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.b_fc_2 = self.add_bias(
            shape=[10],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)

    @tf.function
    def call(self, input, training = None):
        conv_1 = tf.nn.relu(tf.nn.conv2d(input, self.w_conv_1, strides = [1,1,1,1], padding = 'SAME', name = "conv_1"))
        conv_1 = conv_1 + self.b_conv_1
        pooled_1 = tf.nn.max_pool(conv_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = "pool_1")

        conv_2 = tf.nn.relu(tf.nn.conv2d(pooled_1, self.w_conv_2, strides=[1, 1, 1, 1], padding='SAME', name="conv_2"))
        conv_2 = conv_2 + self.b_conv_2
        pooled_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool_2")

        conv_3 = tf.nn.relu(tf.nn.conv2d(pooled_2, self.w_conv_3, strides=[1, 1, 1, 1], padding='SAME', name="conv_3"))
        conv_3 = conv_3 + self.b_conv_3
        pooled_3 = tf.nn.max_pool(conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool_3")

        flattened = tf.reshape(pooled_3, [-1, 4 * 4 * 128], name="Flatten")
        fc_1 = tf.matmul(flattened, self.w_fc_1) + self.b_fc_1
        fc_2 = tf.matmul(fc_1, self.w_fc_2) + self.b_fc_2

with tf.name_scope("Fully_Connected"):
    flattened = tf.reshape(conv_3_pooled, [-1, 4*4*128], name = "Flatten")
    fc_1 = fully_connected(flattened, 1024, name = "Fully_Connected_Layer_1")
    dropout_1 = tf.nn.dropout(fc_1, rate = 1-hold_prob)

with tf.name_scope("Output"):
    prediction = fully_connected(dropout_1, 10, name = "raw_pred")
    prediction = tf.nn.softmax(prediction)



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

@tf.function
def conv2d(input, filter, name):
    conv_name = name +"_Conv"
    with tf.name_scope("Conv"):
        output = tf.nn.conv2d(input, filter, strides = [1,1,1,1], padding = 'SAME', name = conv_name) #this essentially does the filter operation keeping the dimensions the same
        return output

@tf.function
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




def Big_Train():

    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    for i in range(501):

        data, label = datafeeder.nextBatchTrain(1)
        output = model(data, 0.6)
        print(var_list)
        output = np.reshape(output, newshape=(10))
        label = np.reshape(label, newshape=(10))

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

