import pickle
import numpy as np
from Utility import Utility
util = Utility()
import random

class DataStructure:
    def __init__(self, dom, non, overlap, ehistory, history, middle, motion):
        self.dom = dom
        self.non = non
        self.edge_history = ehistory
        self.history = history
        self.middle = middle
        self.motion = motion
        self.overlap = overlap
    def get_dom(self):
        return self.dom
    def get_non(self):
        return self.non
    def get_overlap(self):
        return self.overlap
    def get_ehistory(self):
        return self.edge_history
    def get_history(self):
        return self.history
    def get_middle(self):
        return self.middle
    def get_motion(self):
        return self.motion

class Prep():
    def __init__(self, test_number, requests_list):
        self.trainCount =0
        self.test_number = test_number
        self.requests_list = requests_list

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            objects = pickle.load(fo, encoding = 'bytes')
        return objects

    #preconditions: none
    #postconditions: outputs the 2nd batch as an extracted file
    def unzip_train(self):
        try:
            big_list = self.unpickle("../LINKED/Storage/Data/experimental/SignLanguageData")
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
            carrier_list = list()
            for element in self.requests_list:
                if element == "Motion":
                    carrier = np.asarray(k.get_motion()/255).reshape(100, 100, 1)
                    carrier_list.append(carrier)
                elif element == "History":
                    carrier = np.asarray(k.get_history() / 255).reshape(100, 100, 1) #just to work with the CNN
                    carrier_list.append(carrier)
                elif element == "Middle":
                    carrier = np.asarray(k.get_middle() / 255).reshape(100, 100, 1)  # just to work with the CNN
                    carrier_list.append(carrier)
                elif element == "Overlap":
                    carrier = np.asarray(k.get_overlap() / 255).reshape(100, 100, 1)  # just to work with the CNN
                    carrier_list.append(carrier)
                elif element == "Edge_History":
                    carrier = np.asarray(k.get_ehistory() / 255).reshape(100, 100, 1)  # just to work with the CNN
                    carrier_list.append(carrier)
                else:
                    raise Exception("Data Request not valid. Your options are: {}".format(["History", "Middle", "Overlap", "Motion", "Edge_History"]))
            img_list.append(carrier_list) #now, it's [TRAINLENGTH X # images X 96 X 96 X 1]

        img_list = np.asarray(img_list)

        label_list_dom = np.asarray(label_list_dom) #[TRAINLENGTH X LABELLENGTH X 1]

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
        self.image_list, self.dom = self.unzip_train()
        self.image_list = np.float32(self.image_list)

    def nextBatchTrain_dom(self, batchNum):
        modulus = len(self.image_list)
        image_ = self.image_list[self.trainCount: self.trainCount+batchNum]
        dom_ = self.dom[self.trainCount: self.trainCount+batchNum]
        self.trainCount += batchNum
        self.trainCount = self.trainCount % modulus
        image_list = np.transpose(image_, [1, 0, 2, 3,
                                           4])  # now, it's [# images X TRAINLENGTH X 96 X 96 X 1] This is easier to extract
        return image_list, dom_

    def nextBatchTest_dom(self):
        image_list, dom = self.unzip_test()
        image_list = np.float32(image_list)
        return image_list, dom