from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
hold_prob = 0.6

class CustomLayer(tf.keras.layers.Layer): #this uses a keras layer structure but with a custom layer
    def __init__(self):
        super(CustomLayer, self).__init__()
        self.w_conv_1 = self.add_weight(
            shape=[4,4,3,32],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)
        self.b_conv_1 = self.add_weight(
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
        self.b_conv_2 = self.add_weight(
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
        self.b_conv_3 = self.add_weight(
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
        self.b_fc_1 = self.add_weight(
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
        self.b_fc_2 = self.add_weight(
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
        fc_1_d = tf.nn.dropout(fc_1, rate = 1-hold_prob)
        fc_2 = tf.matmul(fc_1_d, self.w_fc_2) + self.b_fc_2
        prediction = tf.nn.softmax(fc_2)
        return prediction



def Big_Train():

    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
    model= CustomLayer()

    print(len(model.trainable_weights))
    for i in range(501):

        data, label = datafeeder.nextBatchTrain(1)
        output = model.call(data, 0.6)
        with tf.GradientTape() as tape:
            loss = loss_function(y_true = label, y_pred = output)
            print(loss)
            grads = tape.gradient(loss, model.trainable_weights)
            print(grads)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(loss.numpy())



def main():
    print("---the model is starting-----")
    query = input("What mode do you want? Train (t) or Confusion Matrix (m)?\n")
    if query == "t":
        Big_Train()


if __name__ == '__main__':
    main()

