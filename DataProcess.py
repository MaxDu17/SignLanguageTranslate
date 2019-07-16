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
    def __init__(self):
        self.trainCount =0
        self.shuffle_status = False

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            objects = pickle.load(fo, encoding = 'bytes')
        return objects

    #preconditions: none
    #postconditions: outputs the 2nd batch as an extracted file
    def unzip_train(self, shuffle_):
        try:
            big_list = self.unpickle("./SignLanguageData")
        except:
            try:
                big_list = self.unpickle("../LINKED/Storage/Data/SignLanguageData")
            except:
                raise Exception("You big dummy--you forgot to plug in the data drive!")

        if shuffle_:
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

    #preconditions: labels must be in range 0-9
    #postconditions: outputs a 2d array with of 1-hot encodings, with the 1st index being for image
    def Hot_Vec(self, selection): #we have removed a case for the 999 null status, we will add them later
        one_hot_dict, _, size = util.get_dictionaries()
        carrier = np.zeros([size])
        for k in selection:
            mapping = one_hot_dict[k] #so that none of the nodes are sparse
            carrier[mapping] = 1

        return carrier

    def nextBatchTrain(self, batchNum): #not functional at the moment
        image, dom = self.unzip_train(self.shuffle_status)
        self.shuffle_status = False
        modulus = len(image)
        image = image[self.trainCount: self.trainCount+batchNum]
        dom = dom[self.trainCount: self.trainCount+batchNum]
        self.trainCount += batchNum

        if self.trainCount >= modulus:
            self.shuffle_status = True

        self.trainCount = self.trainCount % modulus
        return image, dom

    def nextBatchTrain_dom(self, batchNum):
        image, dom = self.unzip_train(self.shuffle_status)
        self.shuffle_status = False
        modulus = len(image)
        image = image[self.trainCount: self.trainCount+batchNum]
        dom = dom[self.trainCount: self.trainCount+batchNum]
        self.trainCount += batchNum
        print(self.trainCount)

        if self.trainCount >= modulus:
            self.shuffle_status = True

        self.trainCount = self.trainCount % modulus

        return image, dom

    def nextBatchTrain_dom_all(self):
        self.shuffle_status = True
        image, dom = self.unzip_train(self.shuffle_status)
        return image, dom
