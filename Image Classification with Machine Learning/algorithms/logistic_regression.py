"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """

        return 1/(1+np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        
        y_train_ohe = np.zeros((y_train.size, y_train.max() + 1))
        y_train_ohe[np.arange(y_train.size), y_train] = 1
        
        for i in range(self.epochs):
        
            grad = (1/N)*(np.matmul((self.sigmoid(np.matmul(X_train,self.w.T)) - y_train_ohe).T,X_train)) + self.weight_decay*self.w

            self.w = self.w - self.lr*grad
        
        return self.w


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        
        return np.argmax(self.sigmoid(np.matmul(X_test,self.w.T)),axis=1)
