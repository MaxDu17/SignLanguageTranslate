import csv
from collections import Counter
import numpy as np

big_list = list()
non_aug = list()
k = open("DATASET_Motion/dom_labels.csv", "r")
dom = list(csv.reader(k))

w = open("augmentations.csv", "r")
aug = list(csv.reader(w))

for i in range(len(dom)):
    aug_num = int(aug[i][0])
    for j in range(aug_num):
        big_list.extend(dom[i]) #kind of a dumb way to do this, but it works!
    non_aug.extend(dom[i])

print(Counter(big_list))
print(Counter(non_aug))
#this no longer is frequency analysis
k = list(Counter(big_list).keys())
k = [int(j) for j in k]
k.sort()

one_hot_dict= dict() #so this will take the values (keys) and map them to array elements (sequential)

i = 0
for element in k:
    one_hot_dict[element] = i
    i = i + 1

print(one_hot_dict)

look_up_dict= dict() #so this is the inverse mapping

i = 0
for element in k:
    look_up_dict[i] = element
    i = i + 1
