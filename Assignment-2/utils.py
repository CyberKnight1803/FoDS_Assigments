import numpy as np 
import matplotlib.pyplot as plt

def train_test_split(X, y, seed=42):
        np.random.seed(seed)
        p = np.random.permutation(X.shape[1])
        X = X[:, p]
        y = y[p]

        m = X.shape[1]
        training_size = int(m * 0.7)

        X_train = X[:, :training_size]
        X_test = X[:, training_size:]

        y_train = y[:training_size, :]
        y_test = y[training_size:, :]

        return X_train, X_test, y_train, y_test

def standardize(X_train, X_test):
        mean = np.mean(X_train, axis=1, keepdims=True)
        sigma = np.std(X_train, axis=1, keepdims=True)

        X_train = (X_train - mean) / sigma 
        X_test = (X_test - mean) / sigma 

        return X_train, X_test 

def standardize_targets(y_train, y_test):
    mean = np.mean(y_train, axis=0, keepdims=True)
    sigma = np.std(y_train, axis=0, keepdims=True)

    y_train = (y_train - mean) / sigma
    y_test = (y_test - mean) / sigma

    return y_train, y_test
        
def MSE_degree_plot(MSE, dataset_type, GD):
    plt.plot(range(len(MSE)), MSE, 'b-o')
    plt.xlabel('Degrees')
    plt.ylabel('MSE')
    plt.title(f'{dataset_type} MSE vs Degrees using {GD}')
    plt.grid()
