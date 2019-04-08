import keras.backend as K
from tensorflow import count_nonzero

def smoothness(y):
    n_left = K.gather(y, K.arange(0, 2099))
    n_mid = K.gather(y, K.arange(1, 2100))
    n_right = K.gather(y, K.arange(2, 2101))
    result = K.std(n_mid - (n_left + n_mid + n_right) / 3)
    return result

def inflection(y):
    n_left = K.gather(y, K.arange(0, 2099))
    n_mid = K.gather(y, K.arange(1, 2100))
    n_right = K.gather(y, K.arange(2, 2101))
    result = K.cast(count_nonzero(((n_mid > n_left) & (n_mid < n_right)) | ((n_mid < n_left) & (n_mid > n_right))), 'float')
    return result