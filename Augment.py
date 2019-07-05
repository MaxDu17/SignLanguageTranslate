from Utility import Utility
import csv


tool = Utility()

basepath = "DATASET_Motion/"
k = open(basepath + "labels.csv", 'r')
labels = list(csv.reader(k))

for i in range(len(labels)):
    img_path = basepath + str(i) + ".jpg"
    matrix = tool.load_image_to_mat(img_path)
