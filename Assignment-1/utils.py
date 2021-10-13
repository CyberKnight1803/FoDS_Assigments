import numpy as np 

def polynomial_features(X, degree=2, include_bias=True):
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

    if include_bias :
        features = np.insert(features, 0, 1)

    return features

def polynomialFeatures(X, degree=2, include_bias=True):
    m = X.shape[0]
    new_features = []

    for i in range(m):
        x = polynomial_features(X[i], degree, include_bias=True)
        new_features.append(x)
    
    new_features = np.array(new_features)
    
    return new_features.reshape(new_features.shape[1], -1)


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
 
        
