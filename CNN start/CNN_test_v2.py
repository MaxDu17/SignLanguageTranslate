from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
hold_prob = 1

class Convolve(tf.keras.layers.Layer): #this uses a keras layer structure but with a custom layer
    def __init__(self, shape, *args, **kwargs):
        super(Convolve, self).__init__(*args, **kwargs)
        self.shape = shape

    def build(self, input_shape):
        self.w_conv_1 = self.add_weight(
            shape=self.shape,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(),
            #regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name="Convolve")

        self.b_conv_1 = self.add_weight(
            shape=self.shape[3],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            #regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name = "Convolve_Bias")

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
            #regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True,
            name="Fully_Connected_Weight")
        self.b_fc_1 = self.add_weight(
            shape=self.shape[1],
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(),
            #regularizer=tf.keras.regularizers.l2(0.02),
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
    return counter/len(pred)
def Big_Train():

    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
    model = tf.keras.Sequential([Convolve([4, 4, 3, 32]), Convolve([4, 4, 32, 64]), Convolve([4, 4, 64, 128]),
                                 Flatten([-1, 4 * 4 * 128]), FC([4 * 4 * 128, 1024]), FC([1024, 10]), Softmax([])])

    model.build(input_shape = [None, 32, 32, 3])
    print(model.summary())
    print("loading dataset")
    datafeeder.load_train_to_RAM()
    print("starting training")
    for epoch in range(1000):
        data, label = datafeeder.nextBatchTrain(50)
        data = np.float32(data)
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            pred_loss = loss_function(label, predictions)
            if epoch%20 ==0:
                print("***********************")
                print("Finished epoch", epoch)
                print(accuracy(predictions, label))
                print(np.asarray(pred_loss))
                print("***********************")
        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #print("Finished epoch", epoch)
    model.save_weights("Graphs_and_Results/best_weights.h5")


def Conf_mat():
    datafeeder = Prep()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = tf.keras.Sequential([Convolve([4, 4, 3, 32]), Convolve([4, 4, 32, 64]), Convolve([4, 4, 64, 128]),
                                 Flatten([-1, 4 * 4 * 128]), FC([4 * 4 * 128, 1024]), FC([1024, 10]), Softmax([])])

    model.build(input_shape=[None, 32, 32, 3])
    print(model.summary())
    model.load_weights("Graphs_and_Results/best_weights.h5")
    datafeeder = Prep()

    data, label = datafeeder.nextBatchTest()
    data = np.float32(data)
    predictions = model(data, training=True)

    assert len(label) == len(predictions)

    conf = np.zeros(shape = [len(data), len(predictions)])
    for i in range(len(predictions)):
        k = np.argmax(predictions[i])
        l = np.argmax(label[i])
        conf[k][l] += 1
    test = open("Graphs_and_Results/confusion.csv", "w")
    logger = csv.writer(test, lineterminator="\n")

    for iterate in conf:
        logger.writerow(iterate)

    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))
    print(conf)





def main():
    print("---the model is starting-----")
    query = input("What mode do you want? Train (t) or Confusion Matrix (m)?\n")
    if query == "t":
        Big_Train()
    if query == "m":
        Conf_mat()


if __name__ == '__main__':
    main()

