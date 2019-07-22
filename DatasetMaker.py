import pickle
import csv
from Utility import Utility
#this turns the images and labels into one pickle file as DataStructure objects
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

tool = Utility()
dom_reader = list(csv.reader(open("../LINKED/Storage/Data/BIG/dom_labels.csv")))
non_reader = list(csv.reader(open("../LINKED/Storage/Data/BIG/non_labels.csv")))

big_list = list()
basepath = "../LINKED/Storage/Data/BIG/"
path_list = ["augmented_overlap/", "augmented_edge_history/", "augmented_history/", "augmented_middle/",
             "augmented_motion/"]
print(len(dom_reader))
input("Press enter to start dumping to pickle")
for i in range(len(dom_reader)):
    print(i)
    dom = [int(k) for k in dom_reader[i]]
    non = [int(k) for k in non_reader[i]]
    for j in range(200):
        try:
            path_over = basepath + path_list[0] + str(i) + "_" + str(j) + ".jpg"
            path_ehist = basepath + path_list[1] + str(i) + "_" + str(j) + ".jpg"
            path_hist = basepath + path_list[2] + str(i) + "_" + str(j) + ".jpg"
            path_middle = basepath + path_list[3] + str(i) + "_" + str(j) + ".jpg"
            path_motion = basepath + path_list[4] + str(i) + "_" + str(j) + ".jpg"

            matrix_over = tool.load_image_to_mat(path_over)
            matrix_ehist = tool.load_image_to_mat(path_ehist)
            matrix_hist = tool.load_image_to_mat(path_hist)
            matrix_middle = tool.load_image_to_mat(path_middle)
            matrix_motion = tool.load_image_to_mat(path_motion)

            data_object = DataStructure(dom, non, matrix_over, matrix_ehist, matrix_hist, matrix_middle, matrix_motion)
            big_list.append(data_object)
        except FileNotFoundError:
            print("\tProcessed {} files".format(j))
            break

random.shuffle(big_list)
dbfile = open("../LINKED/Storage/Data/BIG/SignLanguageData.pkl", "ab") #extension is important
pickle.dump(big_list, dbfile)
