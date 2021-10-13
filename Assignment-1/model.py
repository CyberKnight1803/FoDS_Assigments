import numpy as np 
from gradientDescent import GD_Variants
import matplotlib.pyplot as plt 

class PolynomialRegression():
    def __init__(self, num_features, learning_rate, n_iters):

        self.n = num_features 
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Initializing parameters
        self.W = np.zeros((num_features, 1))
        self.b = 0

    def compute_cost(self, X, y):
        """
        Arguments
            X = training inputs of shape (n_x, m)
            y = output data
        Returns
            J = Loss  
        """

        m = X.shape[1]                                             # No. of training examples
        y_p = np.dot(self.W.T, X) + self.b
        J = (1 / (2 * m)) * np.sum(np.square(y_p.T - y))
        return J

    def update_params(self, X, y, learning_rate = 0.1, n_iters = 100):
        """
        Arguments
            X = training inputs of shape (n_x, m)
            y = output data
            learning_rate = learning rate 
            n_iters = epochs of the model
        """
        J_history = []
        m = X.shape[1]

        for i in range(n_iters):
            y_p = np.dot(self.W.T, X) + self.b

            self.dW = (1 / m) * np.dot(X, y_p.T - y)
            self.db = (1 / m) * np.sum(y_p.T - y)
            J_history.append(self.compute_cost(X, y))

            # Update
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db

        return J_history
    
    

    def plot_costHistory(self, cost_history):
        plt.plot(cost_history, color='blue')
        plt.xlabel('Cost')
        plt.ylabel('No. of Iterations')
        plt.show()

    def train(self, X, y):

        cost_history = self.update_params(X, y, self.learning_rate, self.n_iters)
        self.plot_costHistory(cost_history)