import csv
from collections import Counter

big_list = list()
non_aug = list()
k = open("dom_labels.csv", "r")
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