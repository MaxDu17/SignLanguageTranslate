import pickle
import numpy as np
from Utility import Utility
util = Utility()
import random

class DataStructure: #this is for the pickle's use
    def __init__(self, dom, non, data):
        self.dom = dom
        self.non = non
        self.data = data
        _, _, self.size = util.get_dictionaries() #just extracting the sizes for later

    def get_dom(self):
        return self.dom

    def get_non(self):
        return self.non

    def get_data(self):
        return self.data

class Prep():
    def __init__(self, test_number):
        self.trainCount =0
        self.test_number = test_number

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            objects = pickle.load(fo, encoding = 'bytes')
        return objects

    #preconditions: none
    #postconditions: outputs the 2nd batch as an extracted file
    def unzip_train(self):
        try:
            big_list = self.unpickle("./SignLanguageData")
        except:
            try:
                big_list = self.unpickle("../LINKED/Storage/Data/SignLanguageData")
            except:
                raise Exception("You big dummy--you forgot to plug in the data drive!")
        test_spot = len(big_list) - self.test_number
        big_list = big_list[0:test_spot]
        random.shuffle(big_list) #this is so we don't get repeats
        print("######SHUFFLING DATASET#######")

        label_list_dom = list()
        img_list = list()
        for k in big_list:
            label_list_dom.append(self.Hot_Vec(k.get_dom()))
            img_list.append(k.get_data()/255)  #remember to normalize
        img_list = np.asarray(img_list)
        label_list_dom = np.asarray(label_list_dom)


        img_list = img_list.reshape(len(big_list), 96, 96, 1) #ignore the pycharm warning here
        return img_list, label_list_dom

    # preconditions: none
    # postconditions: outputs the 2nd batch as an extracted file

    def unzip_test(self):
        try:
            big_list = self.unpickle("./SignLanguageData")
        except:
            try:
                big_list = self.unpickle("../LINKED/Storage/Data/SignLanguageData")
            except:
                raise Exception("You big dummy--you forgot to plug in the data drive!")
        test_spot = len(big_list) - self.test_number
        big_list = big_list[test_spot:] #this is the difference
        random.shuffle(big_list)  # this is so we don't get repeats
        print("######SHUFFLING DATASET#######")

        label_list_dom = list()
        img_list = list()
        for k in big_list:
            label_list_dom.append(self.Hot_Vec(k.get_dom()))
            img_list.append(k.get_data() / 255)  # remember to normalize
        img_list = np.asarray(img_list)
        label_list_dom = np.asarray(label_list_dom)

        img_list = img_list.reshape(len(big_list), 96, 96, 1)  # ignore the pycharm warning here
        return img_list, label_list_dom

    #preconditions: labels must be in range 0-9
    #postconditions: outputs a 2d array with of 1-hot encodings, with the 1st index being for image
    def Hot_Vec(self, selection): #we have removed a case for the 999 null status, we will add them later
        one_hot_dict, _, size = util.get_dictionaries()
        carrier = np.zeros([size])
        for k in selection:
            mapping = one_hot_dict[k] #so that none of the nodes are sparse
            carrier[mapping] = 1

        return carrier

    def load_train_to_RAM(self):
        self.image, self.dom = self.unzip_train()

    def nextBatchTrain_dom(self, batchNum):
        modulus = len(self.image)
        image_ = self.image[self.trainCount: self.trainCount+batchNum]
        dom_ = self.dom[self.trainCount: self.trainCount+batchNum]
        self.trainCount += batchNum
        self.trainCount = self.trainCount % modulus
        return image_, dom_

    def nextBatchTest_dom(self):
        image, dom = self.unzip_test()
        return image, dom