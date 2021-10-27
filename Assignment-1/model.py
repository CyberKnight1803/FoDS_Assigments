import numpy as np 
from gradientDescent import GD_Variants
import matplotlib.pyplot as plt 

class PolynomialRegression():
    def __init__(self, num_features, degree=1, learning_rate=0.01, epochs=1000, GD='BatchGD'):
        self.n = num_features 
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.GD_type = GD_Variants[GD](learning_rate)
        self.costs = []

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
        m = X.shape[1] 
        if self.degree:                                            # No. of training examples
            y_p = np.dot(self.W.T, X) + self.b
        else:
            y_p = self.b
        
        J = (1 / (2 * m)) * np.sum(np.square(y_p.T - y))
        return J
        
    def update_params(self, X, y):
        """
        Arguments
            X = training inputs of shape (n_x, m)
            y = output data
            learning_rate = learning rate 
            n_iters = epochs of the model
        """
        
        for epoch in range(self.epochs):
            self.GD_type(X, y, self, epoch)

        return self.costs 
            
    

    def plot_costHistory(self, cost_history):
        plt.plot(cost_history, color='blue')
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()

    def train(self, X, y):
        cost_history = self.update_params(X, y)
        self.plot_costHistory(cost_history)

    def evaluate(self, X, y):
        if self.degree:
            y_p = np.dot(self.W.T, X) + self.b 
        else:
            y_p = self.b 
        
        cost = self.compute_cost(X, y)
        return cost 
    