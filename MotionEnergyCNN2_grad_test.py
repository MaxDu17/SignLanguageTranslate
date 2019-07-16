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


class Convolve(tf.keras.layers.Layer):  # this uses a keras layer structure but with a custom layer
    def __init__(self, shape, *args, **kwargs):
        super(Convolve, self).__init__(*args, **kwargs)
        self.shape = shape

    def build(self, input_shape):
        self.w_conv_1 = self.add_weight(
            shape=self.shape,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            # regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name="Convolve")

        self.b_conv_1 = self.add_weight(
            shape=self.shape[3],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            # regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name="Convolve_Bias")

    @tf.function
    def call(self, input, training=None):
        conv_1 = tf.nn.relu(tf.nn.conv2d(input, self.w_conv_1, strides=[1, 1, 1, 1], padding='SAME', name="conv_1"))
        conv_1 = conv_1 + self.b_conv_1
        pooled_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool_1")
        return pooled_1


class Flatten(tf.keras.layers.Layer):
    def __init__(self, shape, *args, **kwargs):
        super(Flatten, self).__init__(*args, **kwargs)
        self.shape = shape

    def build(self, input_shape):
        self.size = self.shape
        pass

    @tf.function
    def call(self, input, training=None):
        flattened = tf.reshape(input, self.size, name="Flatten")
        return flattened


class Softmax(tf.keras.layers.Layer):  # this uses a keras layer structure but with a custom layer
    def __init__(self, *args, **kwargs):
        super(Softmax, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, input, training=None):
        prediction = tf.nn.softmax(input)
        return prediction


class FC(tf.keras.layers.Layer):  # this uses a keras layer structure but with a custom layer
    def __init__(self, shape, *args, **kwargs):
        super(FC, self).__init__(*args, **kwargs)
        self.shape = shape

    def build(self, input_shape):
        self.w_fc_1 = self.add_weight(
            shape=self.shape,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            # regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name="Fully_Connected_Weight")
        self.b_fc_1 = self.add_weight(
            shape=self.shape[1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            # regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name="Fully_Connected_Bias")

    @tf.function
    def call(self, input, training=None):
        fc_1 = tf.matmul(input, self.w_fc_1) + self.b_fc_1
        fc_1 = tf.nn.dropout(fc_1, rate=1 - hold_prob)
        return fc_1


def accuracy(pred, labels):
    assert len(pred) == len(labels), "lengths of prediction and labels are not the same"
    counter = 0
    for i in range(len(pred)):
        k = np.argmax(pred[i])
        l = np.argmax(labels[i])
        if k == l:
            counter += 1
    return counter / len(pred)

def Big_Train():
    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    print("*****************Training*****************")
    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = tf.keras.Sequential([Convolve([4, 4, 1, 32]), Convolve([4, 4, 32, 64]),Convolve([4, 4, 64, 128]),
                                 Flatten([-1, 12 * 12 * 128]),FC([12*12*128, 560]),FC([560, output_size]),
                                 Softmax([])])

    model.build(input_shape=[None, 96, 96, 1])
    print(model.summary())
    data, label = datafeeder.nextBatchTrain(100)
    data = np.float32(data)
    for epoch in range(100):
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            pred_loss = loss_function(label, predictions)
            print(accuracy(predictions, label))
        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # print("Finished epoch", epoch)
    model.save_weights("Graphs_and_Results/best_weights.h5")


def Conf_mat():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    inputs = tf.keras.Input(shape=[32, 32, 3])

    x = Convolve([4, 4, 3, 32])(inputs)
    x = Convolve([4, 4, 32, 64])(x)
    x = Convolve([4, 4, 64, 128])(x)
    x = Flatten([-1, 4 * 4 * 128])(x)
    x = FC([4 * 4 * 128, 1024])(x)
    x = FC([1024, 10])(x)
    outputs = Softmax([])(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    model.load_weights("Graphs_and_Results/Version1/best_weights.h5")
    datafeeder = Prep()

    data, label = datafeeder.nextBatchTest()

    acc = model.evaluate(data, label, batch_size=100)
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


