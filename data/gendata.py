import numpy as np

def genFromUnif (n, dim, a, b):
    return np.random.uniform(low = a, high = b, size = (n, dim))
