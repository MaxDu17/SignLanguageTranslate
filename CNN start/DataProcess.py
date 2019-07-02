import pickle
import numpy as np
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

    def augment(self, batch, O_H):
        # we can expand this set from 50000 to 500000 with 10 transformations
        # the two rotations, the two reflections, and the added noise for now, up down, left right, normal
        new_batch = list()
        new_key = list()
        for i in range(len(batch)):
            for k in range(10): #a dumb way to get the data you want
                new_batch.append(O_H[i])
            new_batch.append(batch[i])
            new_batch.append(util.add_noise(batch[i]))
            new_batch.append(util.rot_ck(batch[i]))
            new_batch.append(util.rot_cck(batch[i]))
            new_batch.append(util.flip_lr(batch[i]))
            new_batch.append(util.flip_ud(batch[i]))
            new_batch.append(util.trans_vert(batch[i], -1))
            new_batch.append(util.trans_vert(batch[i], 1))
            new_batch.append(util.trans_hor(batch[i], -1))
            new_batch.append(util.trans_hor(batch[i], 1))
            print(i)
        return new_batch, new_key


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

    def nextBatchTrain(self, batchNum, large):
        batch, O_H = self.unzip_training()
        modulus = 50000
        if large:
            batch, O_H = self.augment(batch, O_H)
            modulus = 500000
        batch = batch[self.trainCount: self.trainCount+batchNum]
        O_H = O_H[self.trainCount: self.trainCount+batchNum]
        self.trainCount += batchNum
        self.trainCount = self.trainCount % modulus
        return batch, O_H


    def nextBatchTest(self):
        batch, O_H = self.unzip_test_small()
        return batch, O_H

    def nextBatchTest_ConfMat(self):
        batch, O_H = self.unzip_test()
        return batch, O_H

    def test_Aug(self): #this is more of a private behind-the-hood function that tests the util library
        batch, _ = self.nextBatchTest()
        batch = batch[0]
        batch = batch * 255
        k = util.trans_vert(batch, 10)
        print(np.shape(k))
        util.display_image(k)

k = Prep()
batch, O_H = k.unzip_training()
batch, O_H = k.augment(batch, O_H)
print(np.shape(batch))

