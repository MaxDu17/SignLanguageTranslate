import pickle
import numpy as np
import csv
from Utility import Utility
util = Utility()
class DataStructure:
    def __init__(self, dom, non, data):
        self.dom = dom
        self.non = non
        self.data = data
    def get_dom(self):
        return self.dom
    def get_non(self):
        return self.non
    def get_data(self):
        return self.data

class Prep():
    def __init__(self):
        self.trainCount =0

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            objects = pickle.load(fo, encoding = 'bytes')
        return objects

    #preconditions: none
    #postconditions: outputs the 2nd batch as an extracted file
    def unzip_train(self):
        big_list = self.unpickle("SignLanguageData")
        label_list_dom = list()
        label_list_non = list()
        img_list = list()

        for k in big_list:
            label_list_dom.append(k.get_dom())
            label_list_non.append(k.get_non())
            img_list.append(k.get_data()/255)  #remember to normalize


    def unzip_test(self):
        files = "test_batch"
        bigdata = list()
        bigdata.append(self.unpickle("data/" + files))
        batch = np.vstack([d[b'data'] for d in bigdata])
        batch = batch / 255
        training_length = len(batch)
        batch = batch.reshape(training_length, 3,32,32).transpose(0,2,3,1)
        labels = np.hstack([d[b'labels'] for d in bigdata])
        O_H = self.oneHot(labels)
        return batch[0:2000], O_H[0:2000]

    def unzip_test_small(self):
        files = "test_batch"
        bigdata = list()
        bigdata.append(self.unpickle("data/" + files))
        batch = np.vstack([d[b'data'] for d in bigdata])
        batch = batch / 255
        training_length = len(batch)
        batch = batch.reshape(training_length, 3,32,32).transpose(0,2,3,1)
        labels = np.hstack([d[b'labels'] for d in bigdata])
        O_H = self.oneHot(labels)
        return batch[0:100], O_H[0:100]


    def getkey(self, data):
        print(data)

    #preconditions: labels must be in range 0-9
    #postconditions: outputs a 2d array with of 1-hot encodings, with the 1st index being for image
    def Hot_Vec(self, selection):
        dimensions = 83
        carrier = np.zeros([dimensions])
        for k in selection:
            carrier[k] = 1
        return carrier

    def nextBatchTrain(self, batchNum):

        modulus = 4580
        batch = batch[self.trainCount: self.trainCount+batchNum]
        O_H = O_H[self.trainCount: self.trainCount+batchNum]
        self.trainCount += batchNum
        self.trainCount = self.trainCount % modulus
        return batch, O_H

k = Prep()
k.unzip_train()

