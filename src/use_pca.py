from sklearn.decomposition import PCA

def use_pca(X):
    pca = PCA(n_components=2)
    pca.fix(X)
    return pca.components_