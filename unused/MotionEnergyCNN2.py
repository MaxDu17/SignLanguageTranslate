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
TEST_AMOUNT = 100

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
    return float(counter)/len(pred)

def Big_Train():
    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    print("*****************Training*****************")
    datafeeder = Prep(TEST_AMOUNT, ["History"])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = tf.keras.Sequential([Convolve([3, 3, 1, 4]), Convolve([3, 3, 4, 8]),
                                 Flatten([-1,25 * 25 * 8]),FC([25*25*8, output_size]),
                                 Softmax([])]) #this declares the layers

    model.build(input_shape=[None, 100, 100, 1]) #this builds the network
    print(model.summary()) #this is more for user reference
    print("loading dataset")
    datafeeder.load_train_to_RAM() #loads the training data to RAM
    summary_writer = tf.summary.create_file_writer(logdir="Graphs_and_Results/")
    print("starting training")

    for epoch in range(501):
        data, label = datafeeder.nextBatchTrain_dom(150)
        data = data[0] #thsi is because we now have multiple images in the pickle
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            pred_loss = loss_function(label, predictions)
            if epoch % 20 == 0 and epoch > 1:
                print("***********************")
                print("Finished epoch", epoch)
                print(accuracy(predictions, label))
                print(np.asarray(pred_loss))
                print("***********************")
                '''
                print(predictions[0])
                print(label[0])
                input("----------------")
                '''
                with summary_writer.as_default():
                    tf.summary.scalar(name = "Loss", data = pred_loss, step = 1)
                    tf.summary.scalar(name = "Accuracy", data = accuracy(predictions, label), step = 1)
                    for var in model.trainable_variables:
                        name = str(var.name)
                        tf.summary.histogram(name = "Variable_" + name, data = var, step = 1)
                    tf.summary.flush()

            if epoch % 100 == 0 and epoch > 1:
                model.save_weights("Graphs_and_Results/best_weights.h5")

        gradients = tape.gradient(pred_loss, model.trainable_variables)
        #print(gradients[1])

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    model.save_weights("Graphs_and_Results/best_weights.h5")
    Test_live(model)

def Test_live(model):
    datafeeder = Prep(TEST_AMOUNT, ["History"])

    data, label = datafeeder.nextBatchTest_dom()
    data = data[0]  # this is because we now have multiple images in the pickle
    predictions = model(data, training=False)

    assert len(label) == len(predictions)
    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))

def Test():
    model = tf.keras.Sequential([Convolve([3, 3, 1, 4]), Convolve([3, 3, 4, 8]),
                                 Flatten([-1, 25 * 25 * 8]), FC([25 * 25 * 8, output_size]),
                                 Softmax([])])  # this declares the layers

    model.build(input_shape=[None, 100, 100, 1])  # this builds the network
    print(model.summary())
    model.load_weights("Graphs_and_Results/best_weights.h5")
    datafeeder = Prep(TEST_AMOUNT,["History"])

    data, label = datafeeder.nextBatchTest_dom()
    data = data[0]  # thsi is because we now have multiple images in the pickle
    predictions = model(data, training=False)

    assert len(label) == len(predictions)
    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))


def main():
    print("---the model is starting-----")
    query = input("What mode do you want? Train (t) or Confusion Matrix (m)?\n")
    if query == "t":
        Big_Train()
        print("###########NOW TESTING##############")
    if query == "m":
        Test()


if __name__ == '__main__':
    main()


