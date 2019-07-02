import csv
import numpy as np
import matplotlib.pyplot as plt
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

    def display_image(self, matrix): #this prints out a 3d image
        images_plot = matrix.astype('uint8')
        plt.imshow(images_plot)
        plt.show()

    def flip_lr(self, matrix):
         return np.fliplr(matrix)

    def flip_ud(self, matrix):
         return np.flipud(matrix)

    def rot_ck(self, matrix):
        return np.rot90(matrix, k = 3)

    def rot_cck(self, matrix):
        return np.rot90(matrix, k = 1)

    def add_noise(self, matrix):
        shape = np.shape(matrix)
        noise = np.random.randint(10, size=shape, dtype='uint8')
        print(noise)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                for k in range(len(matrix[0][0])):
                    if (matrix[i][j][k] != 255):
                        matrix[i][j][k] += noise[i][j][k]

        return matrix

    def trans_right(self, matrix_, amount):
        matrix = np.rot90(matrix_)
        for i in range(len(matrix)-amount):
            matrix[i] = matrix[i+amount]
        matrix = np.rot90(matrix, k = 3)
        return matrix


    def trans_left(self, matrix_, amount):
        matrix = np.rot90(matrix_)
        for i in range(len(matrix)-amount, amount-1, -1):
            matrix[i] = matrix[i-amount]
        matrix = np.rot90(matrix, k=3)
        return matrix


    def trans_up(self, matrix, amount):
        for i in range(len(matrix)-amount):
            matrix[i] = matrix[i+amount]
        return matrix


    def trans_down(self, matrix, amount):
        for i in range(len(matrix)-amount, amount-1, -1):
            matrix[i] = matrix[i-amount]
        return matrix