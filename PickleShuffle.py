import pickle
from DataProcess import DataStructure
import random


def unpickle(file):  # low-level extractor
    with open(file, 'rb') as fo:
        objects = pickle.load(fo, encoding='bytes')
    return objects

print("reading pickle file")
objects = unpickle("../LINKED/Storage/Data/BIG/SignLanguageData.pkl")

random.shuffle(objects)
objects_shuff = objects

print("Dumping shuffled file to disk")
dbfile = open("../LINKED/Storage/Data/BIG/SignLanguageData.pkl", "ab")
pickle.dump(objects_shuff, dbfile)

print("reading pickle file")
objects = unpickle("../LINKED/Storage/Data/BIG/SignLanguageData.pkl")
for k in range(10):
    print(objects[k].get_dom())