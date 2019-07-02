#this code take a confusion matrix csv and plots it
import matplotlib.pyplot as plt
import csv

test = open("Graphs_and_Results/confusion.csv", "r")
logger = csv.reader(test)
matrix = list(logger)
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        matrix[i][j] = float(matrix[i][j])

fig=plt.figure()
ax = fig.add_subplot(111)
color = ax.matshow(matrix)
fig.colorbar(color)
print(matrix)
plt.show()

