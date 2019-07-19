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

big_list = list()
class Convolve():  # this uses a keras layer structure but with a custom layer
    def __init__(self, current_list, shape, name):
        self.current_list = current_list
        self.shape = shape
        self.name = name

    def build(self): #input shape is NOT the parameter you feed into convolve's constructor
        self.w_conv_1 = tf.Variable(initial_value = tf.random.truncated_normal(self.shape, stddev =  0.1,),
                                    name = self.name + "_weight", trainable = True)
        self.current_list.append(self.w_conv_1)

        self.b_conv_1 = tf.Variable(initial_value = tf.zeros(self.shape[3]),
                                    name=self.name + "_bias", trainable = True)
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
        self.w_fc_1 = tf.Variable(initial_value = tf.random.truncated_normal(self.shape, stddev=0.1),
                                  name=self.name + "_weight", trainable = True)
        self.current_list.append(self.w_fc_1)
        self.b_fc_1 = tf.Variable(initial_value = tf.zeros(self.shape[1]),
                                  name=self.name + "_bias", trainable = True)
        self.current_list.append(self.b_fc_1)


    def call(self, input):
        fc_1 = tf.matmul(input, self.w_fc_1) + self.b_fc_1
        #fc_1 = tf.nn.dropout(fc_1, rate=1 - hold_prob) removing this for diagnostic purposes
        return fc_1

class Model():
    def __init__(self):
        self.cnn_1 = Convolve(big_list, [3, 3, 1, 4], "Layer_1_CNN")
        self.cnn_2 = Convolve(big_list, [3, 3, 4, 8], "Layer_2_CNN")

        self.flat = Flatten([-1, 25*25*8], "Fully_Connected")
        self.fc_1 = FC(big_list, [25*25*8, output_size], "Layer_1_FC")
        self.softmax = Softmax()
    def build_model(self):
        self.cnn_1.build()
        self.cnn_2.build()
        self.fc_1.build()
    @tf.function
    def call(self, input):
        x = self.cnn_1.call(input)
        x = self.cnn_2.call(x)
        x = self.flat.call(x)
        x = self.fc_1.call(x)
        output = self.softmax.call(x)
        return output

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
    datafeeder = Prep(150, ["Motion"])
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    print("loading dataset")
    datafeeder.load_train_to_RAM()  # loads the training data to RAM
    summary_writer = tf.summary.create_file_writer(logdir="Graphs_and_Results/")
    print("starting training")

    print("Making model")
    model = Model()
    model.build_model()

    for epoch in range(501):
        data, label = datafeeder.nextBatchTrain_dom(150)
        data = data[0]
        with tf.GradientTape() as tape:
            predictions = model.call(data) #this is the big call

            pred_loss = loss_function(label, predictions) #this is the loss function
            if epoch % 20 == 0 and epoch > 1:
                print("***********************")
                print("Finished epoch", epoch)
                print(accuracy(predictions, label))
                print(np.asarray(pred_loss))
                print("***********************")
                with summary_writer.as_default(): #this is not working rn but I will fix it
                    tf.summary.scalar(name = "Loss", data = pred_loss, step = 1)
                    tf.summary.scalar(name = "Accuracy", data = accuracy(predictions, label), step = 1)
                    for var in big_list:
                        name = str(var.name)
                        tf.summary.histogram(name = "Variable_" + name, data = var, step = 1)
                    tf.summary.flush()

            if epoch % 100 == 0 and epoch > 1:
                #tf.saved_model.save(model.call, "Graphs_and_Results/")

        gradients = tape.gradient(pred_loss, big_list)

        optimizer.apply_gradients(zip(gradients, big_list))


def Test_live(model):
    datafeeder = Prep(TEST_AMOUNT, ["History"])

    data, label = datafeeder.nextBatchTest_dom()
    data = data[0]  # this is because we now have multiple images in the pickle
    predictions = model(data, training=False)

    assert len(label) == len(predictions)
    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))

def Test():
    datafeeder = Prep(150, ["Motion"])
    datafeeder.load_train_to_RAM()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    model = Model()
    model.build_model()
    raise Exception("This section is not ready yet")
    loaded = tf.saved_model.load("Graphs_and_Results/")
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



