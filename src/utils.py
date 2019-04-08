import numpy as np

def generate_noise():
    return np.random.normal(loc = 0.0, scale = 0.05 * np.random.rand(), size = 2101)