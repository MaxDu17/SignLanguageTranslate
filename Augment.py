from Utility import Utility
import csv
import numpy as np
import random
from collections import Counter

tool = Utility()
#VERY IMPORTANT NOTE--THIS IS ONLY FOR DOMINANT HAND
basepath = "DATASET_Motion/"
savepath = "../LINKED/Storage/Data/DATASET_augmented/"
k = open("DATASET_Motion/dom_labels.csv", 'r')
labels = list(csv.reader(k))

#this just find the frequency distribution of the csv
big_list = list()
for k in labels:
    big_list.extend(k)
counted = Counter(big_list)

#we shrink, crop, and augment the dataset
#this is for dom, only

documentation_ = open("augmentations.csv", "w")
documentation = csv.writer(documentation_, lineterminator = "\n")
for i in range(len(labels)):
    img_path = basepath + str(i) + ".jpg"
    matrix = tool.load_image_to_mat(img_path)
    matrix = np.asarray(matrix)
    matrix = tool.resize_image(matrix, 96, 96, "L")

    #this section simply takes the known augmentation factor and takes an average (because there are more than one
    #per row sometimes
    aug_num = int(round((200/counted[str(labels[i][0])])))

    ################
    #here is some programming debauchery
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

    if aug_num == 4:
        tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")
        matrix_LR = tool.flip_lr(matrix)
        tool.save_image(matrix=matrix_LR, path=savepath + str(i) + "_1.jpg", type="L")
        matrix_UD = tool.flip_ud(matrix)
        tool.save_image(matrix=matrix_UD, path=savepath + str(i) + "_2.jpg", type="L")
        matrix_CK = tool.rot_ck(matrix)
        tool.save_image(matrix=matrix_CK, path=savepath + str(i) + "_3.jpg", type="L")

    if aug_num == 5:
        tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")
        matrix_LR = tool.flip_lr(matrix)
        tool.save_image(matrix=matrix_LR, path=savepath + str(i) + "_1.jpg", type="L")
        matrix_UD = tool.flip_ud(matrix)
        tool.save_image(matrix=matrix_UD, path=savepath + str(i) + "_2.jpg", type="L")
        matrix_CK = tool.rot_ck(matrix)
        tool.save_image(matrix=matrix_CK, path=savepath + str(i) + "_3.jpg", type="L")
        matrix_CCK = tool.rot_cck(matrix)
        tool.save_image(matrix=matrix_CCK, path=savepath + str(i) + "_4.jpg", type="L")

    else:
        left = aug_num - 5
        tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")
        matrix_LR = tool.flip_lr(matrix)
        tool.save_image(matrix=matrix_LR, path=savepath + str(i) + "_1.jpg", type="L")
        matrix_UD = tool.flip_ud(matrix)
        tool.save_image(matrix=matrix_UD, path=savepath + str(i) + "_2.jpg", type="L")
        matrix_CK = tool.rot_ck(matrix)
        tool.save_image(matrix=matrix_CK, path=savepath + str(i) + "_3.jpg", type="L")
        matrix_CCK = tool.rot_cck(matrix)
        tool.save_image(matrix=matrix_CCK, path=savepath + str(i) + "_4.jpg", type="L")

        for j in range(left):
            choice = random.randint(1,3)
            if choice == 1:
                matrix_noise = tool.add_noise_L(matrix)
                tool.save_image(matrix=matrix_noise, path=savepath + str(i) + "_" + str(j+5) + ".jpg", type="L")
            elif choice == 2:
                matrix_right = tool.trans_hor(matrix, random.randint(-10, 10), "L")
                tool.save_image(matrix=matrix_right, path=savepath + str(i) + "_" + str(j+5) + ".jpg", type="L")
            else:
                matrix_left = tool.trans_vert(matrix, random.randint(-10, 10), "L")
                tool.save_image(matrix=matrix_left, path=savepath +  str(i) + "_" + str(j+5) + ".jpg", type="L")



