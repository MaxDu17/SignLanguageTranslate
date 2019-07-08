import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
class Utility(): #this class will help with software stuff

    #preconditions: file_object must be a csv file
    #postconditions: returns a matrix containing the contents
    def cast_csv_to_float(self, file_object): #this takes a file object csv and returns a matrix
        logger = csv.reader(file_object)
        matrix = list(logger)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = float(matrix[i][j])
        return matrix

    def cast_csv_to_int(self, file_object):
        logger = csv.reader(file_object)
        matrix = list(logger)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = int(matrix[i][j])
        return matrix
    
    def display_image(self, matrix): #this prints out a 3d image
        images_plot = matrix.astype('uint8')
        plt.imshow(images_plot)
        plt.show()

    def resize_image(self, matrix, width, height, type):
        img = Image.fromarray(matrix.astype(np.uint8), type)
        img = img.resize((width, height))
        return np.asarray(img)

    def save_image(self, matrix, path, type):
        img = Image.fromarray(matrix.astype(np.uint8), type)
        img.save(path)

    def load_image_to_mat(self, path):
        pic = Image.open(path)
        return np.asarray(pic)

    def flip_lr(self, matrix):
         return np.fliplr(matrix)

    def flip_ud(self, matrix):
         return np.flipud(matrix)

    def rot_ck(self, matrix):
        return np.rot90(matrix, k = 3)

    def rot_cck(self, matrix):
        return np.rot90(matrix, k = 1)

    def add_noise_RGB(self, matrix):
        shape = np.shape(matrix)
        noise = np.random.randint(10, size=shape, dtype='uint8')
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                for k in range(len(matrix[0][0])):
                    if (matrix[i][j][k] != 255):
                        matrix[i][j][k] += noise[i][j][k]

        return matrix

    def add_noise_L(self, matrix): #this is for greyscale
        shape = np.shape(matrix)
        carrier = matrix.copy()
        noise = np.random.randint(10, size=shape, dtype='uint8')
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if (matrix[i][j]!= 255):
                    carrier[i][j] += noise[i][j]
        return matrix


    def trans_vert(self, matrix, amount, type):
        img = Image.fromarray(matrix.astype(np.uint8), type)
        a = 1
        b = 0
        c = 0  # left/right (i.e. 5/-5)
        d = 0
        e = 1
        f = amount  # up/down (i.e. 5/-5)
        img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
        return np.asarray(img)

    def trans_hor(self, matrix, amount, type):
        img = Image.fromarray(matrix.astype(np.uint8), type)
        a = 1
        b = 0
        c = amount  # left/right (i.e. 5/-5)
        d = 0
        e = 1
        f = 0  # up/down (i.e. 5/-5)
        img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
        return np.asarray(img)