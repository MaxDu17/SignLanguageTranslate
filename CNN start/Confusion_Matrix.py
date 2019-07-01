import tensorflow as tf
from DataProcess import Prep
import numpy as np
import csv
#THIS RUNS WEATHER FORECAST MODELS
process = Prep()


pbfilename = '../Graphs_and_Results/CNN_test.pb'


with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
    input = graph.get_tensor_by_name("placeholders/Input:0")
    output = graph.get_tensor_by_name("output/prediction:0")

with tf.Session(graph=graph) as sess: