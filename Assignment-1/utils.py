import numpy as np 
import matplotlib.pyplot as plt

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
 
        
def MSE_degree_plot(MSE, dataset_type):
    plt.plot(range(len(MSE)), MSE, 'b-o')
    plt.xlabel('Degrees')
    plt.ylabel('MSE')
    plt.title(f'{dataset_type} MSE vs Degrees')
    plt.grid()