#this code take a confusion matrix csv and plots it
import matplotlib.pyplot as plt
from Utility import Utility
Util = Utility()

test = open("Graphs_and_Results/confusion.csv", "r")
matrix = Util.cast_csv_to_float(test)

fig=plt.figure()
ax = fig.add_subplot(111) #111 is an argument in the form of 3 numbers
color = ax.matshow(matrix)
fig.colorbar(color)
plt.show()

