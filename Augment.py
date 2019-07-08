from Utility import Utility
import csv
import numpy as np
import random


tool = Utility()

basepath = "DATASET_Motion/"
savepath = "DATASET_augmented/"
k = open(basepath + "labels.csv", 'r')
labels = list(csv.reader(k))
#we shrink, crop, and augment the dataset
for i in range(len(labels)):
    print("I'm on image: {}".format(i))
    img_path = basepath + str(i) + ".jpg"
    matrix = tool.load_image_to_mat(img_path)
    matrix = [k[80:560] for k in matrix]
    matrix = np.asarray(matrix)

    matrix = tool.resize_image(matrix, 48, 48, "L")
    tool.save_image(matrix=matrix, path=savepath + str(i) + "_0.jpg", type="L")

    print("\tFlipping left-right")
    matrix_LR = tool.flip_lr(matrix)
    tool.save_image(matrix = matrix_LR, path = savepath + str(i) + "_1.jpg", type = "L")

    print("\tFlipping up-down")
    matrix_UD = tool.flip_ud(matrix)
    tool.save_image(matrix=matrix_UD, path=savepath + str(i) + "_2.jpg", type="L")

    print("\trotating clockwise")
    matrix_CK = tool.rot_ck(matrix)
    tool.save_image(matrix=matrix_CK, path=savepath + str(i) + "_3.jpg", type="L")

    print("\trotating counter clockwise")
    matrix_CCK = tool.rot_cck(matrix)
    tool.save_image(matrix=matrix_CCK, path=savepath + str(i) + "_4.jpg", type="L")

    print("\tadding noise")
    matrix_noise = tool.add_noise_L(matrix)
    tool.save_image(matrix=matrix_noise, path=savepath + str(i) + "_5.jpg", type="L")

    print("\tmoving right")
    matrix_right = tool.trans_hor(matrix, random.randint(1, 6), "L")
    tool.save_image(matrix=matrix_right, path=savepath + str(i) + "_6.jpg", type="L")

    print("\tmoving left")
    matrix_left = tool.trans_hor(matrix, -1 * random.randint(1, 6), "L")
    tool.save_image(matrix=matrix_left, path=savepath + str(i) + "_7.jpg", type="L")

    print("\tmoving up")
    matrix_up = tool.trans_vert(matrix, random.randint(1, 6), "L")
    tool.save_image(matrix=matrix_up, path=savepath + str(i) + "_8.jpg", type="L")

    print("\tmoving down")
    matrix_down = tool.trans_vert(matrix, -1 * random.randint(1, 6), "L")
    tool.save_image(matrix=matrix_down, path=savepath + str(i) + "_9.jpg", type="L")


