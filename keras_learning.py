import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")

