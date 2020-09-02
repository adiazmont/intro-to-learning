"""
    This is an example script of the K-nearest-neighbor
    algorithm in the context of machine-learning.

    The use case uses the MINST dataset for digit classification
    based on 28x28 pixel images.

    Author: Alan A. Diaz-Montiel
    email: alan.diaz@connectcentre.ie
"""

import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


def euclidean_distance(row1, row2):
    """
    Calculate the Euclidean distance between two vectors
    :param row1: numpy array with training samples
    :param row2: numpy array with testing samples
    :return: euclidean distance of row2 to with respect to row1
    """
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (int(row1[i]) - int(row2[i])) ** 2
    return np.sqrt(distance)


def nearest_neighbor(x_trainset, y_trainset, unknown, k=1):
    """
    K-nearest-neighbor implementation with euclidean distance
    :param x_trainset: numpy array of training samples
    :param y_trainset: numpy array of training labels
    :param unknown: numpy array of unknown/unlabeled sample
    :param k: number of neighbors to consider in evaluation
    :return: the class/number given to unknown
    """
    # store distances from all samples to the unknown
    distances = []
    for i, sample in enumerate(x_trainset):
        distances.append((y_trainset[i], euclidean_distance(sample, unknown)))

    # sort the distances in ascending order
    distances.sort(key=lambda x: x[1])

    # look only at the k entries on distances
    kn = distances[:k]
    # take all the classes
    classes = [x[0] for x in kn]
    # create a Counter object with the list of classes
    occurence_count = Counter(classes)
    # return the class with more occurences
    return occurence_count.most_common(1)[0][0]


def evaluate_knn(X_train, X_test, y_train, y_test, k=1):
    """
    Create a classification model with customised
    implementation and evaluate the test set.
    :param X_train: numpy array with training samples
    :param X_test: numpy array with testing samples
    :param y_train: numpy array with training labels
    :param y_test: numpy array with training labels
    :param k: number of k neighbors
    :return: print accuracy of model
    """
    # number of samples to be considered in evaluation
    trainset_size = 100
    # number of tests to evaluate
    testset_size = 100

    # keep track of correctly classified digits
    correct = 0
    for i in range(testset_size):
        # classify digit based on our KNN implementation
        estimation = nearest_neighbor(X_train[:trainset_size], y_train[:trainset_size], X_test[i], k=k)
        # retrieve the correct value
        truth = y_test[i]
        if truth == estimation:
            correct += 1

    # compute accuracy
    accuracy = correct / testset_size * 100
    print("Our algorithm correctly classified a number %s / %s times" % (str(correct), str(testset_size)))
    print("Accuracy of our model: %s %%" % str(accuracy))


def sklearn_knn(X_train, X_test, y_train, y_test, k=1):
    """
    Create a classification model with the KNeighborsClassifier
    from sklearn and evaluate the test set.
    :param X_train: numpy array with training samples
    :param X_test: numpy array with testing samples
    :param y_train: numpy array with training labels
    :param y_test: numpy array with training labels
    :param k: number of k neighbors
    :return: print accuracy of model
    """
    # number of samples to be considered in evaluation
    trainset_size = 100
    # number of tests to evaluate
    testset_size = 100

    # create an instance of KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=k)
    # fit/train/create the model with the training data
    model.fit(X_train[:trainset_size], y_train[:trainset_size].ravel())
    # evaluate the model and compute its accuracy
    accuracy = model.score(X_test[:testset_size], y_test[:testset_size]) * 100
    print("Accuracy of sklearn model %s %%" % str(accuracy))


def main():
    # Store image descriptors and labels from dataset
    # X = np.array(pd.read_csv('dataset/images.csv', header=None))
    # y = np.array(pd.read_csv('dataset/labels.csv', header=None))
    X, y = loadlocal_mnist(
        images_path='dataset/train-images.idx3-ubyte',
        labels_path='dataset/train-labels.idx1-ubyte')

    print('Dimensions of the dataset (samples): %s x %s' % (X.shape[0], X.shape[1]))
    print('Dimensions of the dataset (labels): %s' % len(y))

    # Split dataset in training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print('Dimensions of X_train: %s x %s' % (X_train.shape[0], X_train.shape[1]))
    print('Dimensions of y_train: %s' % (len(y_train)))
    print('Dimensions of X_test: %s x %s' % (X_test.shape[0], X_test.shape[1]))
    print('Dimensions of y_test: %s' % (len(y_test)))

    # Evaluate our KNN implementation
    evaluate_knn(X_train, X_test, y_train, y_test, k=1)
    # Evaluate the Sklearn model implementation
    sklearn_knn(X_train, X_test, y_train, y_test, k=1)


if __name__ == '__main__':
    main()
