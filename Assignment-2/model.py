import numpy as np 
from gradientDescent import GD_Variants
import matplotlib.pyplot as plt 

class LinearRegression():
    def __init__(self, num_features, learning_rate=0.01, epochs=1000, GD='BatchGD', regularizer=None, gamma=0.9):
        self.n = num_features 
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.GD_type = GD_Variants[GD](learning_rate)
        self.costs = []
        self.regularizer = regularizer
        self.gamma = gamma
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
        m = X.shape[1]                                            # No. of training examples
        y_p = np.dot(self.W.T, X) + self.b
        
        J = (1 / (2 * m)) * np.sum(np.square(y_p.T - y))

        if self.regularizer == 'L1':
            J += self.gamma * np.sum(np.abs(self.W))

        if self.regularizer == 'L2':
            J += (self.gamma / 2) * (np.sum(np.square(self.W)))
        
        return J
        
    def update_params(self, X, y, print_cost=True):
        """
        Arguments
            X = training inputs of shape (n_x, m)
            y = output data
            learning_rate = learning rate 
            n_iters = epochs of the model
        """
        
        for epoch in range(self.epochs):
            self.GD_type(X, y, self, epoch, print_cost)

        return self.costs 
            
            
    def plot_costHistory(self, cost_history):
        plt.plot(range(self.epochs),cost_history, color='blue')
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()

    def train(self, X, y, print_cost=True, plot_loss_curves=True):
        cost_history = self.update_params(X, y, print_cost)

        if plot_loss_curves:
            self.plot_costHistory(cost_history)

    def evaluate(self, X, y):
        cost = self.compute_cost(X, y)
        return cost 