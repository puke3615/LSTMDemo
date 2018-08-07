from keras.preprocessing import sequence
from keras.optimizers import *
from keras.models import *
from keras.layers import *

import numpy as np

a = sequence.pad_sequences([1, 2], 10)

print(a)

