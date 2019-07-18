from Utility import Utility
import csv
import numpy as np
import random
from collections import Counter

tool = Utility()

def crop(matrix):
    left_start = 130
    left_end = 330
    up_start = 75
    up_end = 275
    new_mat = list()
    for i in range(up_start, up_end):
        new_mat.append(matrix[i][left_start:left_end])
    new_mat = np.asarray(new_mat)
    return new_mat


#VERY IMPORTANT NOTE--THIS IS ONLY FOR DOMINANT HAND
basepath = "../LINKED/Storage/Data/experimental/motion/"
savepath = "../LINKED/Storage/Data/experimental/augmented_motion/"
k = open("../LINKED/Storage/Data/experimental/dom_labels.csv", 'r')
labels = list(csv.reader(k))

#this just find the frequency distribution of the csv
big_list = list()
for k in labels:
    big_list.extend(k)
counted = Counter(big_list)
print(counted)
input("Press enter to continue")

#we shrink, crop, and augment the dataset
#this is for dom, only

documentation_ = open("../LINKED/Storage/Data/experimental/augmentations.csv", "w")
documentation = csv.writer(documentation_, lineterminator = "\n")
for i in range(len(labels)):
    img_path = basepath + str(i) + ".jpg"
    matrix = tool.load_image_to_mat(img_path)
    matrix = np.asarray(matrix)
    matrix = crop(matrix)#this zooms in
    matrix = tool.resize_image(matrix, 100, 100, "L")

    #this section simply takes the known augmentation factor and takes an average (because there are more than one
    #per row sometimes
    aug_num = int(round((30/counted[str(labels[i][0])])))

    ################
    #here is some programming debauchery. Also, I got rid of flipping and made translation better
    print("Image {} needs {} augmentations, doing that now!".format(i, aug_num))
    documentation.writerow([aug_num])
    if aug_num == 1:
        tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")

    if aug_num == 2:
        tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")
        matrix_LR = tool.flip_lr(matrix)
        tool.save_image(matrix=matrix_LR, path=savepath + str(i) + "_1.jpg", type="L")

    if aug_num == 3:
        tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")
        matrix_LR = tool.flip_lr(matrix)
        tool.save_image(matrix=matrix_LR, path=savepath + str(i) + "_1.jpg", type="L")
        matrix_UD = tool.flip_ud(matrix)
        tool.save_image(matrix=matrix_UD, path=savepath + str(i) + "_2.jpg", type="L")

    else:
        left = aug_num - 3
        tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")
        matrix_LR = tool.flip_lr(matrix)
        tool.save_image(matrix=matrix_LR, path=savepath + str(i) + "_1.jpg", type="L")
        matrix_UD = tool.flip_ud(matrix)
        tool.save_image(matrix=matrix_UD, path=savepath + str(i) + "_2.jpg", type="L")


        for j in range(left):

            matrix_ = tool.trans_hor(matrix, random.randint(-5, 5), "L")
            matrix_ = tool.trans_vert(matrix_, random.randint(-5, 5), "L")
            matrix_ = tool.add_noise_L(matrix_)
            tool.save_image(matrix=matrix_, path=savepath +  str(i) + "_" + str(j+3) + ".jpg", type="L")



