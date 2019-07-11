from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
hold_prob = 0.6

class CustomLayer(tf.keras.layers.Layer): #this uses a keras layer structure but with a custom layer
    def __init__(self, *args, **kwargs):
        super(CustomLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.w_conv_1 = self.add_weight(
            shape=[4,4,3,32],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name="a")

        self.b_conv_1 = self.add_weight(
            shape=[32],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "b")
        self.w_conv_2 = self.add_weight(
            shape=[4, 4, 32, 64],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "c")
        self.b_conv_2 = self.add_weight(
            shape=[64],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "d")
        self.w_conv_3 = self.add_weight(
            shape=[4, 4, 64, 128],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "e")
        self.b_conv_3 = self.add_weight(
            shape=[128],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "f")
        self.w_fc_1 = self.add_weight(
            shape=[4*4*128, 1024],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "g")
        self.b_fc_1 = self.add_weight(
            shape=[1024],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "i")
        self.w_fc_2  = self.add_weight(
            shape=[1024,10],
            dtype=tf.float32,
            initializer=tf.keras.initializers.ones(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "j")
        self.b_fc_2 = self.add_weight(
            shape=[10],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "k")

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
        fc_1_d = tf.nn.dropout(fc_1, rate = 1-hold_prob)
        fc_2 = tf.matmul(fc_1_d, self.w_fc_2) + self.b_fc_2
        prediction = tf.nn.softmax(fc_2)
        return prediction



def Big_Train():

    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
    model= tf.keras.Sequential([CustomLayer()])
    model.compile(optimizer = optimizer, loss = loss_function)

    for i in range(5):
        data, label = datafeeder.nextBatchTrain_all()
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='Graphs_and_Results', histogram_freq=1,
                                                     write_graph=True, write_grads=True, update_freq='epoch')
        cp = tf.keras.callbacks.ModelCheckpoint("Graphs_and_Results/current.ckpt", verbose = 1, save_weights_only = True, period = 1)
        model.fit(data, label, batch_size = 100,  epochs = 1, callbacks = [tensorboard, cp])
    model.save_weights("Graphs_and_Results/best.h5")




def Conf_mat():
    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = tf.keras.Sequential([CustomLayer()])
    model.compile(optimizer=optimizer, loss=loss_function)
    model.load_weights("Graphs_and_Results/current.ckpt")
    datafeeder = Prep()

    data, label = datafeeder.nextBatchTest()

    loss, acc = model.evaluate(data, label)
    print(acc)

def main():
    print("---the model is starting-----")
    query = input("What mode do you want? Train (t) or Confusion Matrix (m)?\n")
    if query == "t":
        Big_Train()
    if query == "m":
        Conf_mat()


if __name__ == '__main__':
    main()

