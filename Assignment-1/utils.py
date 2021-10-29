import numpy as np 
import matplotlib.pyplot as plt
from model import PolynomialRegression

def polynomial_features(X, degree=2, include_bias=False):
    features = X.copy()
    prev_chunk = X
    indices = list(range(len(X)))

    for d in range(1, degree):
        new_chunk = []
        for i, v in enumerate(X):
            next_index = len(new_chunk)
            for coef in prev_chunk[indices[i]:]:
                new_chunk.append(v * coef)
            indices[i] = next_index
        features = np.append(features, new_chunk)
        prev_chunk = new_chunk

    if include_bias:
        features = np.insert(features, 0, 1)

    return np.array(features)

def polynomialFeatures(X, degree=2, include_bias=False):
    m = X.shape[0]
    _x = polynomial_features(X[0], degree, include_bias).reshape(-1, 1)
    for i in range(1, m):
        x = polynomial_features(X[i], degree, include_bias).reshape(-1, 1)
        _x = np.hstack((_x, x))
    
    return _x


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
 
        
def MSE_degree_plot(MSE, dataset_type, GD):
    plt.plot(range(len(MSE)), MSE, 'b-o')
    plt.xlabel('Degrees')
    plt.ylabel('MSE')
    plt.title(f'{dataset_type} MSE vs Degrees using {GD}')
    plt.grid()

def plot_RMSE_loglam(X, y, degree_=9, learning_rate=0.001, epochs=1800, GD='BatchGD', regularizer='L2'):
    gammas = np.array([0.001, 0.011, 0.021, 0.039, 0.056, 0.069, 0.077, 0.094, 0.109, 0.125, 0.157, 0.250, 0.369, 0.444, 0.578, 0.696, 0.787, 0.861, 0.912, 0.99])
    ERMS = []
    ERMS_test = []

    X_new = polynomialFeatures(X, degree=degree_)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, 42)
    X_train, X_test = standardize(X_train, X_test) 

    for i in range(gammas.size):
        pr = PolynomialRegression(X_train.shape[0], degree=degree_, learning_rate=learning_rate, epochs=epochs, regularizer=regularizer, gamma=gammas[i])
        pr.train(X_train, y_train, print_cost=False, plot_loss_curves=False)

        test_mse = pr.evaluate(X_test, y_test)
        ERMS_test.append(np.sqrt(2 * test_mse / X_test.shape[1]))
        ERMS.append(np.sqrt(2 * pr.costs[-1] / X_train.shape[1]))
    
    plt.plot(np.log(gammas), ERMS, c='r')
    plt.plot(np.log(gammas), ERMS_test, c='b')