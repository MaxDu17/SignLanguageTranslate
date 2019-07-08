import pickle
import numpy as np
from Utility import Utility
util = Utility()

class DataStructure: #this is for the pickle's use
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
            label_list_dom.append(self.Hot_Vec(k.get_dom()))
            label_list_non.append(self.Hot_Vec(k.get_non()))
            img_list.append(k.get_data()/255)  #remember to normalize
        img_list = np.asarray(img_list)
        label_list_dom = np.asarray(label_list_dom)
        label_list_non = np.asarray(label_list_non)

        img_list = img_list.reshape(len(big_list), 48, 48, 1) #ignore the pycharm warning here
        return img_list, label_list_dom, label_list_non

    #preconditions: labels must be in range 0-9
    #postconditions: outputs a 2d array with of 1-hot encodings, with the 1st index being for image
    def Hot_Vec(self, selection):
        dimensions = 83
        carrier = np.zeros([dimensions])
        for k in selection:
            if (k == 0):
                raise Exception(
                    "Something isn't right--we have a reference to gesture 0 in the data, which shouldn't exist")
            if(k == 999 and len(selection) == 1):
                carrier[0] = 1
            elif(k == 999 and len(selection) != 1):
                raise Exception("Something isn't right--we have a null value and a non-null value in the same cell")
            else:
                carrier[k] = 1
        return carrier

    def nextBatchTrain(self, batchNum):
        image, dom, non = self.unzip_train()
        modulus = len(image)
        image = image[self.trainCount: self.trainCount+batchNum]
        dom = dom[self.trainCount: self.trainCount+batchNum]
        non = non[self.trainCount: self.trainCount + batchNum]
        self.trainCount += batchNum
        self.trainCount = self.trainCount % modulus
        return image, dom, non

'''
k = Prep()
images, dom, non = k.nextBatchTrain(11)
util.display_image(images[0].reshape(48,48)*255)
print(dom[10])
print("-------")
print(non[10])
'''