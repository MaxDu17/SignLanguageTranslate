import csv
import os

files = os.listdir("Graphs_and_Results/wrongs/")
seen = {}
duplicates = list()

for file in files:
    reader = list(csv.reader(open("Graphs_and_Results/wrongs/" + file, "r")))

    for element in reader:
        compare = int(element[0])
        print(compare)
        if compare not in seen:
            seen[compare] = 1
        else:
            if seen[compare] == 10:
                duplicates.append(compare)

            seen[compare] += 1


k = open("Graphs_and_Results/wrongs/intersections.txt", "w")
k.writelines(str(duplicates))
print(duplicates)

