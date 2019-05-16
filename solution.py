import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

COMPONENT_WEIGHT_1 = np.random.dirichlet(np.ones(10), size=1)[0]
COMPONENT_WEIGHT_2 = np.random.dirichlet(np.ones(10), size=1)[0]


def create_data_set():
    first_mixture = []
    second_mixture = []
    for _ in range(100):
        first_mixture_point = [0, 0]
        second_mixture_point = [0, 0]
        for i in range(5):
            first_mixture_point[0] += COMPONENT_WEIGHT_1[i]*np.random.normal(0, 1)
            first_mixture_point[1] += COMPONENT_WEIGHT_1[i]*np.random.normal(-1, 1)

            second_mixture_point[0] += COMPONENT_WEIGHT_2[i]*np.random.normal(0, 1)
            second_mixture_point[1] += COMPONENT_WEIGHT_2[i]*np.random.normal(1, 1)

        first_mixture.append(first_mixture_point)
        second_mixture.append(second_mixture_point)

    return np.array(first_mixture), np.array(second_mixture)


def plot_boundary(first_mixture, second_mixture):
    # plot decision boundary
    # refer http://www.astroml.org/book_figures/chapter9/fig_simple_naivebayes.html
    X = np.concatenate((first_mixture, second_mixture))
    Y = np.array([0]*100 + [1]*100)
    clf = GaussianNB()
    clf.fit(X, Y)
    # predict the classification probabilities on a grid
    xlim = (-1, 8)
    ylim = (-1, 5)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    plt.scatter(first_mixture[:, 0], first_mixture[:, 1], marker='+', label="First Mixture")
    plt.scatter(second_mixture[:, 0], second_mixture[:, 1], marker='o', c='green', label="Second Mixture")
    plt.contour(xx, yy, Z, [0.5], colors='k')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _first_mixture, _second_mixture = create_data_set()
    # print(_first_mixture)
    # print(_second_mixture)
    plot_boundary(_first_mixture, _second_mixture)
