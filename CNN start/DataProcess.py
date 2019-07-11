import pickle
import numpy as np
import csv
from Utility import Utility
util = Utility()
class Prep():
    def __init__(self):
        self.trainCount =0

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding = 'bytes')
        return dict

    #preconditions: none
    #postconditions: outputs the 2nd batch as an extracted file
    def unzip_training(self):
        files = ["data_batch_1", "data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
        bigdata = list()
        for names in files:
            bigdata.append(self.unpickle("data/" + names))
        batch = np.vstack([d[b'data'] for d in bigdata])
        batch = batch/255
        training_length = len(batch)
        batch = batch.reshape(training_length, 3,32,32).transpose(0,2,3,1)
        labels = np.hstack([d[b'labels'] for d in bigdata])
        O_H = self.oneHot(labels)
        return batch, O_H

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
    def oneHot(self, labels):
        dimensions = len(labels)
        carrier = np.zeros([dimensions, 10])
        for k, l in zip(carrier, labels):
            k[l] = 1
        return carrier

    def nextBatchTrain(self, batchNum):
        batch, O_H = self.unzip_training()
        modulus = 50000
        batch = batch[self.trainCount: self.trainCount+batchNum]
        O_H = O_H[self.trainCount: self.trainCount+batchNum]
        self.trainCount += batchNum
        self.trainCount = self.trainCount % modulus
        batch = np.float32(batch)
        return batch, O_H

    def nextBatchTrain_all(self):
        batch, O_H = self.unzip_training()
        return batch, O_H

    def save_augment(self):
        batch, O_H = self.unzip_training()
        batch, O_H = self.augment(batch, O_H)
        PATH = "../../BIG" #only works on linux
        for i in range(len(batch)):
            sub_path = PATH + '/' + str(i) + '.png'
            test = open("../../BIG/all.csv", "w")
            logger = csv.writer(test, lineterminator="\n")
            logger.writerow(O_H[i])
            util.save_image(batch[i], sub_path)
            print("saving: {}".format(i))

    def nextBatchTest(self):
        batch, O_H = self.unzip_test_small()
        return batch, O_H

    def nextBatchTest_ConfMat(self):
        batch, O_H = self.unzip_test()
        return batch, O_H


