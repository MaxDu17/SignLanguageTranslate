import tensorflow as tf
import numpy as np
from Utility import Utility
import pickle
util = Utility()
from DataProcess import DataStructure

_, _, output_size = util.get_dictionaries()


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
        with tf.name_scope("Convolve"):
            conv = tf.nn.relu(tf.nn.conv2d(input, self.w_conv, strides=[1, 1, 1, 1], padding='SAME', name="conv"))
            conv = conv + self.b_conv
            return conv

    def l2loss(self):
        with tf.name_scope("L2Reg"):
            l2 = tf.reduce_sum(tf.abs(self.w_conv))
            return l2

class Pool():
    def call(self, input):
        with tf.name_scope("Pool"):
            pooled = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool")
            return pooled


class Flatten():
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name

    def call(self, input):
        with tf.name_scope("Flatten"):
            flattened = tf.reshape(input, self.shape, name=self.name + "_flatten")
            return flattened


class Dropout():
    def __init__(self, name, hold_prob):
        self.name = name
        self.hold_prob = hold_prob

    def call(self, input):
        with tf.name_scope("Dropout"):
            output = tf.nn.dropout(input, rate=1 - self.hold_prob, name = self.name)
            return output

class Softmax():
    def call(self, input):
        with tf.name_scope("Softmax"):
            prediction = tf.nn.softmax(input)
            return prediction

class Combine_add():
    def call(self, input1, input2):
        with tf.name_scope("Combine_and_add"):
            output = input1 + input2
            return output

class FC():
    def __init__(self, current_list, shape, name):
        self.current_list = current_list
        self.shape = shape
        self.name = name

    def build(self, from_file = False, weights = None):
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
        with tf.name_scope("Fully_Connected_Layer"):
            fc = tf.matmul(input, self.w_fc) + self.b_fc
            return fc



class ResNetChunk(): #this is a "super" model class, and it builds a resnet chunk
    def __init__(self, deep, weight_shape, current_list, name):
        assert deep % 2 == 0, "depth must be an even number"
        self.depth = deep
        self.shape = weight_shape
        self.current_list = current_list
        self.name = name
        self.layer_list = list() #this contains propagation list

    def build_model_from_pickle(self, exclusive_list):
        assert len(exclusive_list) == self.depth * 2, "there seems to be a dimension problem with the pickle list"
        for i in range(self.depth): #this is the init pass
            layer_obj = Convolve(self.current_list, shape=self.shape, name = self.name + "_Resnet_layer_" + str(i))
            layer_obj.build(from_file = True, weights = exclusive_list[2*i:2*i+2])
            self.layer_list.append(layer_obj)

    def build(self):
        for i in range(self.depth): #this is the init pass
            layer_obj = Convolve(self.current_list, shape=self.shape, name = self.name + "_Resnet_layer_" + str(i))
            layer_obj.build()
            self.layer_list.append(layer_obj)

    def call(self, input): #disregard l2 norm for now, but add later
        current_input = input
        for i in range(0, self.depth, 2):
            with tf.name_scope(name = "Resnet_layer_group_" + str(i)):
                residual = current_input
                output_1 = self.layer_list[i].call(input)
                output_2 = self.layer_list[i+1].call(output_1)
                current_input = output_2 + residual #this propagates the network
        return current_input

    def l2loss(self): #wrapper function
        l2loss = 0
        with tf.name_scope("ResNetL2Loss"):
            for layer in self.layer_list:
                l2loss += layer.l2loss()

        return l2loss
    
    
class Inceptionv1Chunk_naive(): #for programming simplicity, this does NOT have pooling yet
    def __init__(self, current_list, name, data_depth):
        self.current_list = current_list
        self.name = name
        self.data_depth = data_depth

    def build_model_from_pickle(self, exclusive_list):
        assert len(exclusive_list) == 3*2, "there seems to be a dimension problem with the pickle list"
        self.one_one = Convolve(self.current_list, shape = [1, 1, self.data_depth, self.data_depth], name = self.name +
                           "_Inception_one")
        self.one_one.build(from_file = True, weights = exclusive_list[0:2])

        self.three_three = Convolve(self.current_list, shape=[3, 3, self.data_depth, self.data_depth], name=self.name +
                                                                                                   "_Inception_three")
        self.three_three.build(from_file=True, weights=exclusive_list[2:4])

        self.five_five = Convolve(self.current_list, shape=[5, 5, self.data_depth, self.data_depth], name=self.name +
                                                                                                       "_Inception_five")
        self.five_five.build(from_file=True, weights=exclusive_list[4:6])


    def build(self):
        self.one_one = Convolve(self.current_list, shape=[1, 1, self.data_depth, self.data_depth], name=self.name +
                                                                                                   "_Inception_one")
        self.one_one.build()

        self.three_three = Convolve(self.current_list, shape=[3, 3, self.data_depth, self.data_depth], name=self.name +
                                                                                                       "_Inception_three")
        self.three_three.build()

        self.five_five = Convolve(self.current_list, shape=[5, 5, self.data_depth, self.data_depth], name=self.name +
                                                                                                     "_Inception_five")
        self.five_five.build()


    def call(self, input):
        with tf.name_scope("Inception_net_prop"):
            one_one_out = self.one_one.call(input)

            three_three_out = self.three_three.call(input)

            five_five_out = self.five_five.call(input)

            output = tf.concat(values = [one_one_out, three_three_out, five_five_out], axis = 3)


        return output

    def l2loss(self):
        with tf.name_scope("Inception_L2"):
            l2 = self.one_one.l2loss()
            l2 += self.three_three.l2loss()
            l2 += self.five_five.l2loss()

            return l2

class Inceptionv1Chunk(): #for programming simplicity, this does NOT have pooling yet
    def __init__(self, current_list, name, data_depth):
        self.current_list = current_list
        self.name = name
        self.data_depth = data_depth

    def build_model_from_pickle(self, exclusive_list):
        assert len(exclusive_list) == 3*2, "there seems to be a dimension problem with the pickle list"



        self.one_one = Convolve(self.current_list, shape = [1, 1, self.data_depth, self.data_depth], name = self.name +
                           "_Inception_one")
        self.one_one.build(from_file = True, weights = exclusive_list[0:2])

        self.three_three = Convolve(self.current_list, shape=[3, 3, self.data_depth, self.data_depth], name=self.name +
                                                                                                   "_Inception_three")
        self.three_three.build(from_file=True, weights=exclusive_list[2:4])

        self.five_five = Convolve(self.current_list, shape=[5, 5, self.data_depth, self.data_depth], name=self.name +
                                                                                                       "_Inception_five")
        self.five_five.build(from_file=True, weights=exclusive_list[4:6])


    def build(self):
        self.one_one = Convolve(self.current_list, shape=[1, 1, self.data_depth, self.data_depth], name=self.name +
                                                                                                   "_Inception_one")
        self.one_one.build()

        self.three_three = Convolve(self.current_list, shape=[3, 3, self.data_depth, self.data_depth], name=self.name +
                                                                                                       "_Inception_three")
        self.three_three.build()

        self.five_five = Convolve(self.current_list, shape=[5, 5, self.data_depth, self.data_depth], name=self.name +
                                                                                                     "_Inception_five")
        self.five_five.build()


    def call(self, input):
        with tf.name_scope("Inception_net_prop"):
            one_one_out = self.one_one.call(input)

            three_three_out = self.three_three.call(input)

            five_five_out = self.five_five.call(input)

            output = tf.concat(values = [one_one_out, three_three_out, five_five_out], axis = 3)


        return output

    def l2loss(self):
        with tf.name_scope("Inception_L2"):
            l2 = self.one_one.l2loss()
            l2 += self.three_three.l2loss()
            l2 += self.five_five.l2loss()

            return l2


class Metrics():
    def accuracy(self, pred, labels):
        assert len(pred) == len(labels), "lengths of prediction and labels are not the same"
        counter = 0
        for i in range(len(pred)):
            k = np.argmax(pred[i])
            l = np.argmax(labels[i])
            if k == l:
                counter += 1
        return float(counter) / len(pred)

    def record_error(self, data, labels, pred):
        assert len(data[0]) == len(pred), "your data and prediction don't match"
        assert len(pred) == len(labels), "your prediction and labels don't match"

        wrong = list()
        right = list()
        for i in range(len(data[0])):
            if np.argmax(pred[i]) != np.argmax(labels[i]):
                wrong.append(data[0][i])
            else:
                right.append(data[0][i])
        return right, wrong



