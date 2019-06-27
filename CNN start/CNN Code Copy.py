import pickle
import numpy as np
import matplotlib.pyplot as plt
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

def unzip():
    files = ["batches.meta", "data_batch_1", "data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
    bigdata = [0,1,2,3,4,5,6]
    for i, d in zip(files, bigdata):
        bigdata[d] = unpickle("data/" + i)
    #bigdata is this structure:
    #array for batch, just use one for similar
    #then, the key is the label name, like b'data'
    #you access the data via a dictionary access
    batch_2 = bigdata[3]
    return batch_2

def process(data):
    data.reshape(10000, 3, 32, 32) #this turns into the tensor that it should be in
    return data

def makeplot():
    data = batch