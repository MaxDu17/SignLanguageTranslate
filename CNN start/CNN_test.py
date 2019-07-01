from DataProcess import Prep
import tensorflow as tf
import numpy as np



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
    return(tf.matmul(input, W) + b)

x = tf.placeholder(tf.float32, shape = [None, 32,32,3])
truth = tf.placeholder(tf.float32, shape = [None, 10])
hold_prob = tf.placeholder(tf.float32)

with tf.name_scope("Layer_1"):
    conv_1 = convolutional_layer(x, shape = [4,4,3,32], name = "Layer_1") # 4 and 4 is the window, 3 is the color channels and 32 is number of output layers (filters)
    conv_1_pooled = max_pool(conv_1, name = "Layer_1")

with tf.name_scope("Layer_2"):
    conv_2 = convolutional_layer(conv_1_pooled, shape = [4,4,32,64], name = "Layer_2")
    conv_2_pooled = max_pool(conv_2, name = "Layer_2")

with tf.name_scope("Fully_Connected"):
    flattened = tf.reshape(conv_2_pooled, [-1, 8*8*64], name = "Flatten")
    fc_1 = fully_connected(flattened, 1024, name = "Fully_Connected_Layer_1")
    dropout_1 = tf.nn.dropout(fc_1, keep_prob = hold_prob)

with tf.name_scope("Output"):
    prediction = fully_connected(dropout_1, 10, name = "Fully_Connected_Layer_2")

with tf.name_scope("Loss_and_Optimizer"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = truth, logits = prediction, name = "Softmax_loss"))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, name = "Optimizer")
    train = optimizer.minimize(loss)

with tf.name_scope("Saver"):
    tf.summary.scalar("Loss", loss)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=9)
    tf.summary.image("Training data", x)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    print("I'm starting")
    sess.run( tf.global_variables_initializer())
    writer = tf.summary.FileWriter("Graphs_and_Results/CNN_test",
                                   sess.graph)  # this will write summary tensorboard
    tf.train.write_graph(sess.graph_def,name = "PBTXT", logdir="Graphs_and_Results/graph.pbtxt")
    datafeeder = Prep()

    for i in range(501):
        data, label = datafeeder.nextBatchTrain(100)
        prediction_, loss_, summary, _ = sess.run([prediction, loss, summary_op, train], feed_dict = {x:data, truth:label, hold_prob:1})
        print("Epoch: {}. Loss: {}".format(i, loss_))
        if i % 100== 0 and i > 0:
            saver.save(sess, "Graphs_and_Results/CNN_test", global_step=i)
            writer.add_summary(summary, global_step=i)
            data, label = datafeeder.nextBatchTest()
            correct = 0
            prediction_ = sess.run(prediction, feed_dict = {x:data, truth:label, hold_prob:1})
            for l in range(len(label)):
                if(np.argmax(prediction_[l]) == np.argmax(label[l])):
                    correct +=1
            print("This is the accuracy: {}".format(correct/len(prediction_)))
