from sklearn.datasets import load_iris, load_boston, load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def pca_run(X, y, title):
    X_pca = PCA(n_components=2).fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.title(title + ' PCA')
    plt.show()


def tsne_runs(X, y, title):
    perplexities = [15, 30, 45]
    for sigma in perplexities:
        X_tsne = TSNE(n_components=2, perplexity=sigma).fit_transform(X)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
        plt.title(title + ' TSNE, perplexity=' + str(sigma))
        plt.show()


X, y = load_iris(return_X_y=True)
pca_run(X, y, 'iris flower')
tsne_runs(X, y, 'iris flower')

X, y = load_boston(return_X_y=True)
pca_run(X, y, 'boston housing prices')
tsne_runs(X, y, 'boston housing prices')

X, y = load_wine(return_X_y=True)
pca_run(X, y, 'wine')
tsne_runs(X, y, 'wine')
