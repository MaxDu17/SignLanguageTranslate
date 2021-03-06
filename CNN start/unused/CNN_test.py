from DataProcess import Prep
import tensorflow as tf
import numpy as np
import csv
import os

def make_weights(shape, name):
    weight_name = name + "_Weight"
    with tf.name_scope("Weights"):
        distribution = tf.random.truncated_normal(shape, stddev =  0.1, name = weight_name)
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
    return(tf.matmul(input, W) + b)

with tf.name_scope("Placeholders"):
    x = tf.placeholder(tf.float32, shape = [None, 32,32,3], name = "Input")
    truth = tf.placeholder(tf.float32, shape = [None, 10], name = "Label")
    hold_prob = tf.placeholder(tf.float32)

with tf.name_scope("Layer_1"):
    conv_1 = convolutional_layer(x, shape = [4,4,3,32], name = "Layer_1") # 4 and 4 is the window, 3 is the color channels and 32 is number of output layers (filters)
    conv_1_pooled = max_pool(conv_1, name = "Layer_1")

with tf.name_scope("Layer_2"):
    conv_2 = convolutional_layer(conv_1_pooled, shape = [4,4,32,64], name = "Layer_2")
    conv_2_pooled = max_pool(conv_2, name = "Layer_2")

with tf.name_scope("Layer_3"):
    conv_3 = convolutional_layer(conv_2_pooled, shape=[4, 4, 64, 128], name="Layer_3")
    conv_3_pooled = max_pool(conv_3, name="Layer_3")

with tf.name_scope("Fully_Connected"):
    flattened = tf.reshape(conv_3_pooled, [-1, 4*4*128], name = "Flatten")
    fc_1 = fully_connected(flattened, 1024, name = "Fully_Connected_Layer_1")
    dropout_1 = tf.nn.dropout(fc_1, rate = 1-hold_prob)

with tf.name_scope("Output"):
    prediction = fully_connected(dropout_1, 10, name = "raw_pred")
    prediction_out = tf.multiply(1.0, prediction,
                             name="Prediction")

with tf.name_scope("Loss_and_Optimizer"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = truth, logits = prediction, name = "Softmax_loss"))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, name = "Optimizer")
    train = optimizer.minimize(loss)

with tf.name_scope("Saver"):
    tf.summary.scalar("Loss", loss)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=7)

init = tf.global_variables_initializer()




def Big_Train(sess):
    sess.run(tf.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter("Graphs_and_Results/",
                                             sess.graph)  # this will write summary tensorboard
    datafeeder = Prep()

    display, _ = datafeeder.nextBatchTrain(10)
    tf.compat.v1.summary.image("10 training data examples", display, max_outputs=10)
    for i in range(501):
        data, label = datafeeder.nextBatchTrain(100)
        prediction_, loss_, summary, _ = sess.run([prediction, loss, summary_op, train],
                                                  feed_dict={x: data, truth: label, hold_prob: 1})
        print("Epoch: {}. Loss: {}".format(i, loss_))
        if i % 10 == 0:
            writer.add_summary(summary, global_step=i)
        if i % 100 == 0 and i > 0:
            saver.save(sess, "Graphs_and_Results/CNN_test", global_step=i)
            data, label = datafeeder.nextBatchTest()
            correct = 0
            prediction_ = sess.run(prediction, feed_dict={x: data, truth: label, hold_prob: 1})
            for l in range(len(label)):
                if (np.argmax(prediction_[l]) == np.argmax(label[l])):
                    correct += 1
            print("This is the accuracy: {}".format(correct / len(prediction_)))

def confMat(sess):
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('Graphs_and_Results/'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    datafeeder = Prep()

    data, label = datafeeder.nextBatchTest_ConfMat()

    matrix = np.zeros([10, 10])

    prediction_ = sess.run(prediction, feed_dict={x: data, truth: label, hold_prob: 1})
    for l in range(len(prediction_)):
        k = np.argmax(prediction_[l])
        m = np.argmax(label[l])
        matrix[k][m] += 1
    test = open("Graphs_and_Results/confusion.csv", "w")
    logger = csv.writer(test, lineterminator="\n")

    for iterate in matrix:
        logger.writerow(iterate)
    print(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    print(matrix)


def main():
    with tf.Session() as sess:
        print("---the model is starting-----")
        query = input("What mode do you want? Train (t) or Confusion Matrix (m)?\n")
        if query == "t":
            Big_Train(sess)
        elif query == "m":
            confMat(sess)

if __name__ == '__main__':
    main()

