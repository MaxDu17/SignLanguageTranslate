from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
from Utility import Utility
import pickle
util = Utility()
from DataProcess import DataStructure


hold_prob = 0.8
_, _, output_size = util.get_dictionaries()
TEST_AMOUNT = 100
VALID_AMOUNT = 50

LEARNING_RATE_INIT = 0.001
L2WEIGHT = 0.01

big_list = list()

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
    def call(self, input):
        output = tf.nn.dropout(input, rate=1 - hold_prob)
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


class Model():
    def __init__(self):
        self.cnn_1 = Convolve(big_list, [3, 3, 1, 4], "Layer_1_CNN")
        self.cnn_2 = Convolve(big_list, [3, 3, 4, 4], "Layer_2_CNN")
        self.cnn_3 = Convolve(big_list, [3, 3, 4, 4], "Layer_2_CNN")
        self.pool_1 = Pool()

        self.cnn_4 = Convolve(big_list, [3, 3, 4, 8], "Layer_2_CNN")
        self.pool_2 = Pool()

        self.flat = Flatten([-1, 25*25*8], "Fully_Connected")
        self.fc_1 = FC(big_list, [25*25*8, output_size], "Layer_1_FC")
        self.softmax = Softmax()

    def build_model_from_pickle(self, file_dir):
        big_list = unpickle("Graphs_and_Results/resnet/SAVED_WEIGHTS")
        #weights and biases are arranged alternating and in order of build
        self.cnn_1.build(from_file = True, weights = big_list[0:2])
        self.cnn_2.build(from_file = True, weights = big_list[2:4])
        self.cnn_3.build(from_file=True, weights=big_list[4:6])
        self.cnn_4.build(from_file=True, weights=big_list[6:8])
        self.fc_1.build(from_file = True, weights = big_list[6:8])

    def build_model(self):
        self.cnn_1.build()
        self.cnn_2.build()
        self.cnn_3.build()
        self.cnn_4.build()
        self.fc_1.build()

    @tf.function
    def call(self, input):
        residual = x = self.cnn_1.call(input) #layer 1
        l2 = self.cnn_1.l2loss()

        x = self.cnn_2.call(x) #layer 2
        l2 += self.cnn_2.l2loss()

        x = self.cnn_3.call(x) #layer 3
        l2 += self.cnn_3.l2loss()
        x = x + residual #this is the resnet structure, without pooling

        x = self.pool_1.call(x)

        x = self.cnn_4.call(x) #layer 4
        l2 += self.cnn_4.l2loss()
        x = self.pool_2.call(x)

        x = self.flat.call(x)
        x = self.fc_1.call(x) #fully connected layer
        output = self.softmax.call(x)
        return output, l2

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

    datafeeder = Prep(TEST_AMOUNT, VALID_AMOUNT, ["Motion"])

    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_INIT)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    print("loading dataset")
    datafeeder.load_train_to_RAM()  # loads the training data to RAM
    summary_writer = tf.summary.create_file_writer(logdir="Graphs_and_Results/resnet/")
    print("starting training")

    print("Making model")
    model = Model()
    model.build_model()
    tf.summary.trace_on(graph=True, profiler=True)

    for epoch in range(1001):
        data, label = datafeeder.nextBatchTrain_dom(150)
        data = data[0]
        with tf.GradientTape() as tape:
            predictions, l2_loss = model.call(data) #this is the big call

            pred_loss = loss_function(label, predictions) #this is the loss function
            pred_loss = pred_loss + L2WEIGHT * l2_loss #this implements lasso regularization
            if epoch == 0: #creates graph
                with summary_writer.as_default():
                    tf.summary.trace_export(name="Graph", step=0, profiler_outdir="Graphs_and_Results/resnet")

            if epoch % 20 == 0 and epoch > 1:
                print("***********************")
                print("Finished epoch", epoch)
                print("Accuracy: {}".format(accuracy(predictions, label)))
                print("Loss: {}".format(np.asarray(pred_loss)))
                print("***********************")
                with summary_writer.as_default():
                    tf.summary.scalar(name = "Loss", data = pred_loss, step = epoch)
                    tf.summary.scalar(name = "Accuracy", data = accuracy(predictions, label), step = epoch)
                    for var in big_list:
                        name = str(var.name)
                        tf.summary.histogram(name = "Variable_" + name, data = var, step = epoch)
                    tf.summary.flush()

            if epoch % 50 == 0 and epoch > 1:
                Validation(model, datafeeder)

            if epoch % 100 == 0 and epoch > 1:
                print("\n##############SAVING MODE##############\n")
                try:
                    os.remove("Graphs_and_Results/resnet/SAVED_WEIGHTS.pkl")
                except:
                    print("the saved weights were not removed, because they were not there!")
                dbfile = open("Graphs_and_Results/resnet/SAVED_WEIGHTS.pkl", "ab")
                pickle.dump(big_list, dbfile)

        gradients = tape.gradient(pred_loss, big_list)

        optimizer.apply_gradients(zip(gradients, big_list))
    Test_live(model, datafeeder)

def Validation(model, datafeeder):
    print("\n##############VALIDATION##############\n")

    data, label = datafeeder.GetValid_dom()
    data = data[0]  # this is because we now have multiple images in the pickle
    predictions, l2loss = model.call(data)
    assert len(label) == len(predictions)
    print("This is the validation set accuracy: {}".format(accuracy(predictions, label)))

def Test_live(model, datafeeder):
    print("\n##############TESTING##############\n")

    data, label = datafeeder.GetTest_dom()
    data = data[0]  # this is because we now have multiple images in the pickle
    predictions, l2loss = model.call(data)

    assert len(label) == len(predictions)
    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))

def Test():
    print("Making model")
    model = Model()
    model.build_model_from_pickle("Graphs_and_Results/resnet/SAVED_WEIGHTS")

    datafeeder = Prep(TEST_AMOUNT, VALID_AMOUNT, ["Motion"])
    datafeeder.load_train_to_RAM()
    data, label = datafeeder.GetTest_dom()
    data = data[0]  # this is because we now have multiple images in the pickle
    predictions, l2loss = model.call(data)

    assert len(label) == len(predictions), "something is wrong with the loaded model or labels"
    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))


def main():
    print("Starting the program!")
    query = input("What mode do you want? Train (t) or Test from model (m)?\n")
    if query == "t":
        Big_Train()
    if query == "m":
        Test()


if __name__ == '__main__':
    main()



