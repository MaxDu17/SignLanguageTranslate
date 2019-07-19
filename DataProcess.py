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

class Prep(): #we use a lot of global variables to make thins more universal
    def __init__(self, test_number, valid_number, requests_list):
        self.trainCount =0
        self.test_number = test_number
        self.valid_number = valid_number
        self.requests_list = requests_list
        self.image_list = None
        self.test_list = None

    def unpickle(self, file): #low-level extractor
        with open(file, 'rb') as fo:
            objects = pickle.load(fo, encoding = 'bytes')
        return objects

    #extracts and makes test/train sets
    def unzip_pickle(self): #THIS IS THE EXTRACTION PROGRAM. ONLY CALL ME ONCE!!!
        assert self.test_list == None, "you can't call unzip_train more than once"
        try:
            self.big_list = self.unpickle("../LINKED/Storage/Data/BIG/SignLanguageData")
        except:
            raise Exception("You big dummy--you forgot to plug in the data drive!")
        #|||||||TRAIN||||||||VALID|||||TEST||||

        assert len(self.big_list) > 0, "the data file appears to be empty"
        self.test_list = self.big_list[0:self.test_number] #allocates test set
        self.valid_list = self.big_list[self.test_number:self.test_number + self.valid_number]#allocates validation set
        self.train_list = self.big_list[self.test_number + self.valid_number:]  # allocates training set


    #preconditions: needs a list of DataStructure objects
    #postconditions: extracts, parses, and normalizes image data
    def next_train_list(self): #call me every time you need a fresh set
        print("Retrieving Training Dataset")
        random.shuffle(self.train_list)
        label_list_dom = list()
        img_list = list()
        for k in self.train_list:
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

        self.image_list = np.float32(np.asarray(img_list))
        self.dom = np.asarray(label_list_dom) #[TRAINLENGTH X LABELLENGTH X 1]


    # preconditions: none
    # postconditions: shuffles the test batch, normalizes and extracts
    def next_test_list(self):
        print("Retrieving Test Dataset")
        assert len(self.test_list) > 0, "You haven't executed \"load_train_to_RAM\""
        random.shuffle(self.test_list) #this is so we don't get repeats

        label_list_dom = list()
        img_list = list()
        for k in self.test_list:
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

        self.test_img_list = np.float32(np.asarray(img_list))
        self.test_dom = np.asarray(label_list_dom) #[TRAINLENGTH X LABELLENGTH X 1]


    def next_valid_list(self):
        print("Retrieving Validation Dataset")
        assert len(self.valid_list) > 0, "You haven't executed \"load_train_to_RAM\""
        random.shuffle(self.valid_list) #this is so we don't get repeats

        label_list_dom = list()
        img_list = list()
        for k in self.valid_list:
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

        self.valid_img_list = np.float32(np.asarray(img_list))
        self.valid_dom = np.asarray(label_list_dom) #[TRAINLENGTH X LABELLENGTH X 1]


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
        print("Loading training data to RAM")
        self.unzip_pickle() #extracts pickle files to self.test_list, self.valid_list and self.train_list (training list)
        self.next_train_list() #makes image and dom lists


    def nextBatchTrain_dom(self, batchNum):
        try:
            modulus = len(self.image_list)
        except AttributeError:
            print("You haven't executed \"load_train_to_RAM\"")
            quit()
        image_ = self.image_list[self.trainCount: self.trainCount+batchNum]
        dom_ = self.dom[self.trainCount: self.trainCount+batchNum]
        self.trainCount += batchNum

        if self.trainCount > modulus:
            print("Looping around again")
            self.next_train_list() #this refreshes the lists self.image_list, self.dom by shuffling them

        self.trainCount = self.trainCount % modulus
        image_list = np.transpose(image_, [1, 0, 2, 3,
                                           4])  # now, it's [# images X TRAINLENGTH X 96 X 96 X 1] This is easier to extract
        return image_list, dom_

    def GetTest_dom(self):
        self.next_test_list()

        image_list = np.transpose(self.test_img_list, [1, 0, 2, 3,
                                           4])  # now, it's [# images X TRAINLENGTH X 96 X 96 X 1] This is easier to extract
        test_dom = self.test_dom
        return image_list, test_dom

    def GetValid_dom(self):
        self.next_valid_list()

        image_list = np.transpose(self.valid_img_list, [1, 0, 2, 3,
                                           4])  # now, it's [# images X TRAINLENGTH X 96 X 96 X 1] This is easier to extract
        valid_dom = self.valid_dom
        return image_list, valid_dom