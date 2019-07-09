from Utility import Utility
import csv
import numpy as np
import random
from collections import Counter

tool = Utility()
#VERY IMPORTANT NOTE--THIS IS ONLY FOR DOMINANT HAND
basepath = "DATASET_Motion/"
savepath = "DATASET_augmented/"
k = open("dom_labels.csv", 'r')
labels = list(csv.reader(k))


#we shrink, crop, and augment the dataset


for i in range(len(labels)):
    print("I'm on image: {}".format(i))
    img_path = basepath + str(i) + ".jpg"
    matrix = tool.load_image_to_mat(img_path)
    matrix = np.asarray(matrix)
    matrix = tool.resize_image(matrix, 96, 96, "L")

    tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")

    matrix_LR = tool.flip_lr(matrix)
    tool.save_image(matrix=matrix_LR, path=savepath + str(i) + "_1.jpg", type="L")

    matrix_UD = tool.flip_ud(matrix)
    tool.save_image(matrix=matrix_UD, path=savepath + str(i) + "_2.jpg", type="L")

    matrix_CK = tool.rot_ck(matrix)
    tool.save_image(matrix=matrix_CK, path=savepath + str(i) + "_3.jpg", type="L")

    matrix_CCK = tool.rot_cck(matrix)
    tool.save_image(matrix=matrix_CCK, path=savepath + str(i) + "_4.jpg", type="L")

    matrix_noise = tool.add_noise_L(matrix)
    tool.save_image(matrix=matrix_noise, path=savepath + str(i) + "_5.jpg", type="L")

    matrix_hor = tool.trans_hor(matrix, random.randint(-6, 6), "L")
    tool.save_image(matrix=matrix_hor, path=savepath + str(i) + "_6.jpg", type="L")

    matrix_vert = tool.trans_vert(matrix, random.randint(-6, 6), "L")
    tool.save_image(matrix=matrix_vert, path=savepath +  str(i) + "_7.jpg", type="L")

    matrix_hor = tool.trans_hor(matrix, random.randint(-6, 6), "L")
    tool.save_image(matrix=matrix_hor, path=savepath + str(i) + "_8.jpg", type="L")

    matrix_vert = tool.trans_vert(matrix, random.randint(-6, 6), "L")
    tool.save_image(matrix=matrix_vert, path=savepath + str(i) + "_9.jpg", type="L")



