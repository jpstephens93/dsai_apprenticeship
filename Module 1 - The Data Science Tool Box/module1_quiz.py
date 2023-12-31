import numpy as np
import pandas as pd

X = np.array([[1, 2], [3, 4]])
Y = X.reshape(X.shape[0], -1).T

numbers = {
    'Number': [1, 2, 3],
    'English': ['one', 'two', 'three'],
    'German': ['eins', 'zwei', 'drei']
}
df = pd.DataFrame(numbers)
df.transpose()

ints = [1, 2, 3]
np.array(ints).dtype

sum(range(5), -1)
from numpy import *
sum(range(5), -1)
