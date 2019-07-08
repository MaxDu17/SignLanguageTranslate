import pickle
import csv
from Utility import Utility
#this turns the images and labels into one pickle file as DataStructure objects

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

tool = Utility()
dom_reader = list(csv.reader(open("dom_labels.csv")))
non_reader = list(csv.reader(open("non_labels.csv")))

big_list = list()
basepath = "DATASET_augmented/"
for i in range(len(dom_reader)):
    dom = [int(k) for k in dom_reader[i]]
    non = [int(k) for k in non_reader[i]]
    for j in range(10):
        path = basepath + str(i) + "_" + str(j) + ".jpg"
        matrix = tool.load_image_to_mat(path)
        data_object = DataStructure(dom, non, matrix)
        big_list.append(data_object)

dbfile = open("SignLanguageData", "ab")
pickle.dump(big_list, dbfile)
