from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
from Utility import Utility
import pickle
util = Utility()
from DataProcess import DataStructure
from MyCNNLibrary import * #this is my own "keras" extension onto tensorflow

hold_prob = 0.8
_, _, output_size = util.get_dictionaries()
TEST_AMOUNT = 100
VALID_AMOUNT = 50

LEARNING_RATE_INIT = 0.001
L2WEIGHT = 0.05

big_list = list()
IMAGE = "Motion"
version = 2
class Model():
    def __init__(self):
        self.cnn_init = Convolve(big_list, [3, 3, 1, 4], "Layer_1_CNN")
        self.resNetChunk = ResNetChunk(deep = 6, weight_shape = [3, 3, 4, 4], current_list = big_list)
        self.pool_1 = Pool()
        self.cnn_8 = Convolve(big_list, [3, 3, 4, 8], "Layer_8_CNN")
        self.pool_2 = Pool()

        self.flat = Flatten([-1, 25*25*8], "Fully_Connected")
        self.fc_1 = FC(big_list, [25*25*8, output_size], "Layer_1_FC")
        self.softmax = Softmax()

    def build_model_from_pickle(self, file_dir):
        big_list = unpickle(file_dir)
        #weights and biases are arranged alternating and in order of build
        self.cnn_init.build(from_file = True, weights = big_list[0:2])
        self.resNetChunk.build_model_from_pickle(exclusive_list = big_list[2:6*2+2]) #there are 12 w and b

        self.cnn_8.build(from_file=True, weights=big_list[14:16])
        self.fc_1.build(from_file = True, weights = big_list[16:18])

    def build_model(self):
        self.cnn_init.build()
        self.resNetChunk.build()
        self.cnn_8.build()
        self.fc_1.build()

    @tf.function
    def call(self, input):
        x = self.cnn_init.call(input) #layer 1
        l2loss = self.cnn_init.l2loss()

        x = self.resNetChunk.call(x) #this should roll it all out
        l2loss += self.resNetChunk.l2loss()

        x = self.pool_1.call(x)

        x = self.cnn_8.call(x) #layer 4
        l2loss += self.cnn_8.l2loss()

        x = self.pool_2.call(x)

        x = self.flat.call(x)

        x = self.fc_1.call(x) #fully connected layer
        output = self.softmax.call(x)
        return output, l2loss #we bypass the l2 error for now

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
    try:
        os.mkdir("Graphs_and_Results/inception/" + IMAGE + str(version))
    except:
        pass

    status = tf.test.is_gpu_available()
    print("Is there a GPU available: {}".format(status))

    print("*****************Training*****************")

    datafeeder = Prep(TEST_AMOUNT, VALID_AMOUNT, [IMAGE])

    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_INIT)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    print("loading dataset")
    datafeeder.load_train_to_RAM()  # loads the training data to RAM
    summary_writer = tf.summary.create_file_writer(logdir="Graphs_and_Results/resnet/" + IMAGE + str(version) + "/")
    print("starting training")

    print("Making model")
    model = Model()
    model.build_model()

    train_logger = csv.writer(open("Graphs_and_Results/inception/" + IMAGE + str(version) + "/xentropyloss.csv", "w"),
                              lineterminator="\n")
    acc_logger = csv.writer(open("Graphs_and_Results/inception/" + IMAGE + str(version) + "/accuracy.csv", "w"),
                              lineterminator="\n")
    l2_logger = csv.writer(open("Graphs_and_Results/inception/" + IMAGE + str(version) + "/l2.csv", "w"),
                              lineterminator="\n")
    valid_logger = csv.writer(open("Graphs_and_Results/inception/" + IMAGE + str(version) + "/valid.csv", "w"),
                              lineterminator="\n")


    tf.summary.trace_on(graph=True, profiler=False) #set profiler to true if you want compute history

    for epoch in range(501):
        data, label = datafeeder.nextBatchTrain_dom(150)
        data = data[0]
        with tf.GradientTape() as tape:
            predictions, l2_loss = model.call(data) #this is the big call

            pred_loss_ = loss_function(label, predictions) #this is the loss function
            pred_loss = pred_loss_ + L2WEIGHT * l2_loss

            train_logger.writerow([np.asarray(pred_loss)])
            acc_logger.writerow([accuracy(predictions, label)])
            l2_logger.writerow([np.asarray(l2_loss)])

            if epoch == 0: #creates graph
                with summary_writer.as_default():
                    tf.summary.trace_export(name="Graph", step=0, profiler_outdir="Graphs_and_Results/resnet/" + IMAGE + str(version) + "/")

            print("***********************")
            print("Finished epoch", epoch)
            print("Accuracy: {}".format(accuracy(predictions, label)))
            print("Loss: {}".format(np.asarray(pred_loss)))
            print("L2 Loss: {}".format(np.asarray(l2_loss)))
            print("***********************")

            if epoch % 20 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar(name = "XEntropyLoss", data = pred_loss_, step = epoch)
                    tf.summary.scalar(name="L2Loss", data=l2_loss, step=epoch)
                    tf.summary.scalar(name = "Accuracy", data = accuracy(predictions, label), step = epoch)
                    for var in big_list:
                        name = str(var.name)
                        tf.summary.histogram(name = name, data = var, step = epoch)
                    tf.summary.flush()

            if epoch % 50 == 0:
                valid_accuracy = Validation(model, datafeeder)
                with summary_writer.as_default():
                    tf.summary.scalar(name = "Validation_accuracy", data = valid_accuracy, step = epoch)
                valid_logger.writerow([valid_accuracy])

            if epoch % 100 == 0 and epoch > 1:
                print("\n##############SAVING MODE##############\n")
                try: #because for some reason, the pickle files are incremental
                    os.remove("Graphs_and_Results/resnet/" + IMAGE + str(version) + "/SAVED_WEIGHTS.pkl")
                except:
                    print("the saved weights were not removed because they were not there!")
                dbfile = open("Graphs_and_Results/resnet/" + IMAGE + str(version) + "/SAVED_WEIGHTS.pkl", "ab")

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
    valid_accuracy = accuracy(predictions, label)
    print("This is the validation set accuracy: {}".format(valid_accuracy))
    return valid_accuracy

def Test_live(model, datafeeder):
    print("\n##############TESTING##############\n")

    data, label = datafeeder.GetTest_dom()
    data = data[0]  # this is because we now have multiple images in the pickle
    predictions, l2loss = model.call(data)

    assert len(label) == len(predictions)
    conf = np.zeros(shape=[len(label[0]), len(predictions[0])])
    for i in range(len(predictions)):
        k = np.argmax(predictions[i])
        l = np.argmax(label[i])
        conf[k][l] += 1
    test = open("Graphs_and_Results/resnet/" + IMAGE + str(version) + "/confusion.csv", "w")
    logger = csv.writer(test, lineterminator="\n")

    test_ = open("Graphs_and_Results/resnet/" + IMAGE + str(version) + "/results.csv", "w")
    logger_ = csv.writer(test_, lineterminator="\n")
    logger_.writerow([accuracy(predictions, label)])

    for iterate in conf:
        logger.writerow(iterate)

    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))

def Test():
    print("Making model")
    model = Model()
    model.build_model_from_pickle("Graphs_and_Results/resnet/" + IMAGE + str(version) + "/SAVED_WEIGHTS.pkl")

    datafeeder = Prep(TEST_AMOUNT, VALID_AMOUNT, [IMAGE])
    datafeeder.load_train_to_RAM()
    data, label = datafeeder.GetTest_dom()
    data = data[0]  # this is because we now have multiple images in the pickle
    predictions, l2loss = model.call(data)

    assert len(label) == len(predictions), "something is wrong with the loaded model or labels"
    conf = np.zeros(shape=[len(label[0]), len(predictions[0])])
    for i in range(len(predictions)):
        k = np.argmax(predictions[i])
        l = np.argmax(label[i])
        conf[k][l] += 1
    test = open("Graphs_and_Results/resnet/" + IMAGE + str(version) + "/confusion.csv", "w")
    logger = csv.writer(test, lineterminator="\n")

    test_ = open("Graphs_and_Results/resnet/" + IMAGE + str(version) + "/results.csv", "w")
    logger_ = csv.writer(test_, lineterminator="\n")
    logger_.writerow([accuracy(predictions, label)])

    for iterate in conf:
        logger.writerow(iterate)
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



