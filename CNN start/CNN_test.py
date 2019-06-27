from DataProcess import Prep
import tensorflow as tf
import numpy as np

datafeeder = Prep(3)
matrix, labels = datafeeder.allPrepare()


