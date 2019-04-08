from keras.models import load_model
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from ref.xxy.denoise import *

autoencoder = load_model('autoencoder.h5')
wavelength_num = 2101

def ae(X):
    return autoencoder.predict(X.reshape(1, wavelength_num)).reshape(wavelength_num)

def sg(X):
    return savgol_filter(X, 101, 2)

def dwt(X):
    return thresholding(X)

def pca(X):
    pca = PCA(n_components=2)
    pca.fix(X)
    return pca.components_