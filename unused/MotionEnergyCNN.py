from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os
from DataProcess import DataStructure

def make_weights(shape, name):
    weight_name = name + "_Weight"
    with tf.name_scope("Weights"):
        distribution = tf.truncated_normal(shape, stddev =  0.1, name = weight_name)
        var = tf.Variable(distribution)
        tf.summary.histogram(weight_name, var)
        return var

def make_bias(shape, name):
    bias_name = name + "_Bias"
    with tf.name_scope("Biases"):
        distribution = tf.constant(0.1, shape = shape, name = bias_name)
        var = tf.Variable(distribution)
        tf.summary.histogram(bias_name, var)
        return var

def conv2d(input, filter, name):
    conv_name = name +"_Conv"
    with tf.name_scope("Conv"):
        output = tf.nn.conv2d(input, filter, strides = [1,1,1,1], padding = 'SAME', name = conv_name) #this essentially does the filter operation keeping the dimensions the same
        return output

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
    raw = tf.matmul(input, W) + b
    return(tf.nn.sigmoid(raw))

with tf.name_scope("Placeholders"):
    x = tf.placeholder(tf.float32, shape = [None, 48,48,1], name = "Input")
    dom = tf.placeholder(tf.float32, shape = [None, 83], name = "Label_dom")
    non = tf.placeholder(tf.float32, shape = [None, 83], name = "Label_non")
    hold_prob = tf.placeholder(tf.float32)

with tf.name_scope("Layer_1"):
    conv_1 = convolutional_layer(x, shape = [4,4,1,32], name = "Layer_1") # 4 and 4 is the window, 3 is the color channels and 32 is number of output layers (filters)
    conv_1_pooled = max_pool(conv_1, name = "Layer_1")

with tf.name_scope("Layer_2"):
    conv_2 = convolutional_layer(conv_1_pooled, shape = [4,4,32,64], name = "Layer_2")
    conv_2_pooled = max_pool(conv_2, name = "Layer_2")

with tf.name_scope("Layer_3"):
    conv_3 = convolutional_layer(conv_2_pooled, shape=[4, 4, 64, 128], name="Layer_3")
    conv_3_pooled = max_pool(conv_3, name="Layer_3")

with tf.name_scope("Fully_Connected_DOM"):
    flattened_dom = tf.reshape(conv_3_pooled, [-1, 6*6*128], name = "Flatten_DOM")
    fc_1_dom = fully_connected(flattened_dom, 1152, name = "Fully_Connected_Layer_1_DOM")
    dropout_1_dom = tf.nn.dropout(fc_1_dom, rate = 1-hold_prob)
    fc_2_dom = fully_connected(dropout_1_dom, 576, name="Fully_Connected_Layer_2_DOM")

with tf.name_scope("Fully_Connected_NON"):
    flattened_non = tf.reshape(conv_3_pooled, [-1, 6*6*128], name = "Flatten_NON")
    fc_1_non = fully_connected(flattened_non, 1152, name = "Fully_Connected_Layer_1_NON")
    dropout_1_non = tf.nn.dropout(fc_1_non, rate = 1-hold_prob)
    fc_2_non = fully_connected(dropout_1_non, 576, name="Fully_Connected_Layer_2_NON")

with tf.name_scope("Output"):
    prediction_dom = fully_connected(fc_2_dom, 83, name="raw_pred_DOM")
    prediction_non = fully_connected(fc_2_non, 83, name = "raw_pred_NON")

with tf.name_scope("Loss_and_Optimizer"):
    loss_dom = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=dom, logits=prediction_dom, name="Cross_entropy_loss_DOM"))
    loss_non = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=non, logits=prediction_non, name="Cross_entropy_loss_NON"))

    big_loss = tf.add(loss_dom, loss_non, name = "Loss_Concat")

    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, name = "Optimizer")
    train = optimizer.minimize(big_loss)

with tf.name_scope("Saver"):
    tf.summary.scalar("Loss_dom", loss_dom)
    tf.summary.scalar("Loss_non", loss_non)
    tf.summary.scalar("Loss_ALL", big_loss)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=7)

init = tf.global_variables_initializer()




def Big_Train(sess):
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('Graphs_and_Results/CNNv1/'))
    if ckpt and ckpt.model_checkpoint_path:
        if input("Do you want to restore previous session? (y/n)") == 'y':
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("session discarded")

    writer = tf.compat.v1.summary.FileWriter("Graphs_and_Results/CNNv1/",
                                             sess.graph)  # this will write summary tensorboard
    datafeeder = Prep()

    for i in range(1501):
        data, dom_label, non_label = datafeeder.nextBatchTrain(100)

        prediction_dom_, prediction_non_, loss_dom_, loss_non_, summary, _ = sess.run(
            [prediction_dom, prediction_non, loss_dom, loss_non, summary_op, train],
            feed_dict={x: data, dom: dom_label, non: non_label, hold_prob: 1})

        print("Epoch: {}. Dom_Loss: {}. Non_loss: {}. Big_loss: {}".format(i, loss_dom_, loss_non_, loss_dom_ + loss_non_))
        if i % 10 == 0:
            writer.add_summary(summary, global_step=i)
            print("This is the prediction: {}".format(prediction_non_[0]))
            print("This is the label: {}".format(non_label[0]))
        if i % 100 == 0:
            saver.save(sess, "Graphs_and_Results/CNNv1/Sign", global_step=i)
            #add testing function here


def main():
    with tf.Session() as sess:
        print("---the model is starting-----")
        query = input("What mode do you want? Train (t) or Confusion Matrix (m)?\n")
        if query == "t":
            Big_Train(sess)
        elif query == "m":
            print("Under construction")
            quit()
if __name__ == '__main__':
    main()

