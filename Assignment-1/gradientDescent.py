import numpy as np 

class GD():
    def __init__(self, lRate):
        self.lRate = lRate 
    
    def update(self, model):
        model.W -= self.lRate * model.dW 
        model.b -= self.lRate * model.db
    
class BatchGD(GD):
    def __init__(self, lRate):
        super().__init__(lRate=lRate)
    
    def __call__(self, X, y, model, itr, costs, print_cost=True):
        m = X.shape[1]
        y_p = np.dot(model.W.T, X) + model.b 
        model.dW = (1 / m) * np.dot(X, y_p.T - y)
        model.db = (1 / m) * np.sum(y_p.T - y)

        self.update(model)

        cost = model.compute_cost(X, y)
        costs.append(cost)

        if print_cost and itr % 50 == 0:
            print(f"Cost after iteration {itr + 1} : {cost}")

class stochasticGD(GD):
    def __init__(self, lRate=0.01):
        super().__init__(lRate=lRate)
    
    def __call__(self, X, y, model, costs, itr, print_cost=True):
        m = X.shape[1]
        k = y.shape[0]

        final_epoch_cost = 0
        for i in range(0, m):
            y_p = np.dot(model.W.T, X[:, i].reshape(-1, 1))

            model.dW = np.dot(X[:, i], y_p.T - y[:, i].reshape(k, -1))
            model.db = np.sum(y_p.T - y[:, i].reshape(k, -1))

            self.update(model)

            cost = model.compute_cost(X[:, i].reshape(-1, 1), y[:, i].reshape(k, -1))
            costs.append(cost)
            final_epoch_cost = cost 
        
        if print_cost:
            print(f"Cost after epoch {itr + 1}: {final_epoch_cost}")


GD_Variants = {
    'BatchGD': BatchGD,
    'StochasticGD': stochasticGD
}

