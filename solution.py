import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

COMPONENT_WEIGHT_1 = random.uniform(0, 1)
COMPONENT_WEIGHT_2 = random.uniform(0, 1)


def create_data_set():
    first_mixture = []
    second_mixture = []
    for _ in range(100):
        first_mixture_point = []
        second_mixture_point = []
        for _ in range(5):
            first_mixture_gaussian = COMPONENT_WEIGHT_1*np.random.normal(0, 1) \
                                     + (1-COMPONENT_WEIGHT_2)*np.random.normal(-1, 1)
            first_mixture_point.append(first_mixture_gaussian)

            second_mixture_gaussian = COMPONENT_WEIGHT_2*np.random.normal(0, 1) \
                                      + (1 - COMPONENT_WEIGHT_2)*np.random.normal(1, 1)
            second_mixture_point.append(second_mixture_gaussian)
        first_mixture.append(first_mixture_point)
        second_mixture.append(second_mixture_point)

    return first_mixture, second_mixture


def plot_boundary(first_mixture, second_mixture):
    # Reduce dimension
    pca = PCA(n_components=2)
    first_mixture_reduction = pca.fit_transform(first_mixture)
    second_mixture_reduction = pca.fit_transform(second_mixture)

    # plot decision boundary
    # refer http://www.astroml.org/book_figures/chapter9/fig_simple_naivebayes.html
    X = np.concatenate((first_mixture_reduction, second_mixture_reduction))
    Y = np.array([0]*100 + [1]*100)
    clf = GaussianNB()
    clf.fit(X, Y)
    # predict the classification probabilities on a grid
    xlim = (first_mixture_reduction[:, 0].min()-1, first_mixture_reduction[:, 0].max()+1)
    ylim = (first_mixture_reduction[:, 1].min()-1, first_mixture_reduction[:, 1].max()+1)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    plt.scatter(first_mixture_reduction[:, 0], first_mixture_reduction[:, 1], marker='+', label="First Mixture")
    plt.scatter(second_mixture_reduction[:, 0], second_mixture_reduction[:, 1], marker='o', c='green', label="Second Mixture")
    plt.contour(xx, yy, Z, [0.5], colors='k')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _first_mixture, _second_mixture = create_data_set()
    plot_boundary(_first_mixture, _second_mixture)
