import pickle
import numpy as np
import matplotlib.pyplot as plt
class Prep():
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding = 'bytes')
        return dict

    #preconditions: none
    #postconditions: outputs the 2nd batch as an extracted file
    def unzip(self):
        files = ["data_batch_1", "data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
        bigdata = list()
        for names in files:
            print(names)
            bigdata.append(self.unpickle("data/" + names))

        #bigdata is this structure:
        #array for batch, just use one for similar
        #then, the key is the label name, like b'data'
        #you access the data via a dictionary access
        batch = np.vstack([d[b'data'] for d in bigdata])
        training_length = len(batch)
        batch = batch.reshape(training_length, 3,32,32).transpose(0,2,3,1)
        labels = np.vstack([d[b'labels'] for d in bigdata])
        return batch, labels


    def getkey(self, data):
        print(data)

    #preconditions: this data must be from the process() function
    #postconditions: this plots an image
    def makeplot(self, data, labels, i):
        images_plot = data.astype('uint8')
        print(images_plot.shape)
        plt.imshow(images_plot[i])
        print(labels[i])
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
        matrix, labels = self.unzip()
        matrix = matrix/255
        one_hot = self.oneHot(labels)
        return matrix, one_hot

    #postcondition: returns a certain number of image matrices for testing
    def testPrepare(self):
        matrix, labels = self.unzip()
        matrix = matrix/255
        one_hot = self.oneHot(labels)
        matrix = matrix[0:self.TESTSIZE]
        one_hot = one_hot[0:self.TESTSIZE]
        return matrix, one_hot

    #postconditions: returns testsize for regulations
    def getTest(self):
        return self.TESTSIZE

k = Prep()
k.unzip()