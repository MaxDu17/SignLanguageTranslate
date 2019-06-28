from DataProcess import Prep
import tensorflow as tf
import numpy as np



def make_weights(shape):
    distribution = tf.truncated_normal(shape, stddev =  0.1)
    var = tf.Variable(distribution)
    return var

def make_bias(shape):
    distribution = tf.constant(0.1, shape = shape)
    var = tf.Variable(distribution)
    return var

def conv2d(input, filter):
    output = tf.nn.conv2d(input, filter, strides = [1,1,1,1], padding = 'SAME') #this essentially does the filter operation keeping the dimensions the same
    return output

def max_pool(input):
    output = tf.nn.max_pool(input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') #ksize is the window size, and it just pools every 4 grid into 1 (need to work out why later)
    return output

def convolutional_layer(input, shape):
    W = make_weights(shape)
    b = make_bias([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def fully_connected(input, end_size):
    in_size = int(input.get_shape()[1])
    W = make_weights([in_size, end_size])
    b = make_bias([end_size])
    return(tf.matmul(input, W) + b)

x = tf.placeholder(tf.float32, shape = [None, 32,32,3])
truth = tf.placeholder(tf.float32, shape = [None, 10])
hold_prob = tf.placeholder(tf.float32)

conv_1 = convolutional_layer(x, shape = [4,4,3,32]) # 4 and 4 is the window, 3 is the color channels and 32 is number of output layers (filters)
conv_1_pooled = max_pool(conv_1)
conv_2 = convolutional_layer(conv_1_pooled, shape = [4,4,32,64])
conv_2_pooled = max_pool(conv_2)


flattened = tf.reshape(conv_2_pooled, [-1, 8*8*64])
fc_1 = fully_connected(flattened, 1024)
dropout_1 = tf.nn.dropout(fc_1, keep_prob = hold_prob)
prediction = fully_connected(dropout_1, 10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = truth, logits = prediction))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    print("I'm starting")
    sess.run( tf.global_variables_initializer())
    datafeeder = Prep()

    for i in range(500):
        data, label = datafeeder.nextBatchTrain(100)
        prediction_, loss_, _ = sess.run([prediction, loss, train], feed_dict = {x:data, truth:label, hold_prob:1})

        if i % 500 == 0:
            data, label = datafeeder.nextBatchTest()
            correct = 0
            prediction_ = sess.run(prediction, feed_dict = {x:data, truth:label, hold_prob:1})
            for k, l in zip(prediction_, label):
                if(tf.argmax(k) == tf.argmax(l)):
                    correct += 1
            print("epoch: {}".format(i))
            print("This is the accuracy: {}".format(correct/len(prediction_))
            print("This is the loss: {}".format(loss_))


