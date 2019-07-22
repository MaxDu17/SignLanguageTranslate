import pickle
from DataProcess import DataStructure
import random


def unpickle(file):  # low-level extractor
    with open(file, 'rb') as fo:
        objects = pickle.load(fo, encoding='bytes')
    return objects

print("reading pickle file")
objects = unpickle("../LINKED/Storage/Data/BIG/SignLanguageData")

random.shuffle(objects)

print("Dumping shuffled file to disk")
dbfile = open("../LINKED/Storage/Data/BIG/SignLanguageData", "ab")
pickle.dump(objects, dbfile)