import pickle
import numpy as np
import matplotlib.pyplot as plt
class Prep():
    def __init__(self, databatch):
        self.databatch = databatch
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding = 'bytes')
        return dict

    #preconditions: none
    #postconditions: outputs the 2nd batch as an extracted file
    def unzip(self):
        files = ["batches.meta", "data_batch_1", "data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
        bigdata = [0,1,2,3,4,5,6]
        for i, d in zip(files, bigdata):
            bigdata[d] = self.unpickle("data/" + i)
        #bigdata is this structure:
        #array for batch, just use one for similar
        #then, the key is the label name, like b'data'
        #you access the data via a dictionary access
        batch_2 = bigdata[self.databatch]
        return batch_2

    #preconditions: this data must be from the unzip() function containing 10000 X 3072 array
    #postconditions: this function returns a 10000 (images) x 3 x 32 x 32 (32 x 32 image with 3 color channels)
    #it also returns labels
    def process(self, data):
        matrix = data[b'data']
        labels = data[b'labels']
        matrix = matrix.reshape(10000, 3, 32, 32) #this turns into the tensor that it should be in
        matrix = matrix.transpose(0, 2, 3, 1)
        return matrix, labels

    #preconditions: this data must be from the process() function
    #postconditions: this plots an image
    def makeplot(self, data, i):
        images_plot = data.astype('uint8')
        print(images_plot.shape)
        plt.imshow(images_plot[i])
        plt.show()

    #preconditions: labels must be in range 0-9
    #postconditions: outputs a 2d array with of 1-hot encodings, with the 1st index being for image
    def oneHot(self, labels):
        dimensions = len(labels)
        carrier = np.zeros([dimensions, 10])
        for k, l in zip(carrier, labels):
            k[l] = 1
        return carrier

    #postconditions: returns the data and label (one-hot) arrays
    def allPrepare(self):
        matrix, labels = self.process(self.unzip())
        matrix = matrix/255
        one_hot = self.oneHot(labels)
        return matrix, one_hot
