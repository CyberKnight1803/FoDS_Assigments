import numpy as np 

class GD():
    def __init__(self, lRate):
        self.lRate = lRate 
    
    def update(self, model):
        if model.degree:
            model.W -= self.lRate * model.dW 
            model.b -= self.lRate * model.db
        else:
            model.b -= self.lRate * model.db
    
class BatchGD(GD):
    def __init__(self, lRate):
        super().__init__(lRate=lRate)
    
    def __call__(self, X, y, model, itr, print_cost=True):
        m = X.shape[1]

        if model.degree:
            y_p = np.dot(model.W.T, X) + model.b 
            model.dW = (1 / m) * np.dot(X, y_p.T - y)
            model.db = (1 / m) * np.sum(y_p.T - y)
        else:
            y_p = np.array([model.b]).reshape(-1, 1) 
            model.db = (1 / m) * np.sum(y_p.T - y)

        self.update(model)

        cost = model.compute_cost(X, y)
        model.costs.append(cost)

        if print_cost and (itr + 1) % 50 == 0:
            print(f"Cost after iteration {itr + 1} : {cost}")

class StochasticGD(GD):
    def __init__(self, lRate=0.01):
        super().__init__(lRate=lRate)
    
    def __call__(self, X, y, model, itr, print_cost=True):
        m = X.shape[1]

        final_epoch_cost = 0
        for i in range(0, m):

            if model.degree:
                y_p = np.dot(model.W.T, X[:, i].reshape(-1, 1))
                model.dW = np.dot(X[:, i].reshape(-1, 1), y_p.T - y[i, :].reshape(1, -1))
                model.db = np.sum(y_p.T - y[i, :].reshape(1, -1))
            else:
                y_p = np.array(model.b).reshape(-1, 1)
                model.db = np.sum(y_p.T - y[i, :]).reshape(-1, 1)
            self.update(model)

            cost = model.compute_cost(X[:, i].reshape(-1, 1), y[i, :].reshape(1, -1))
            model.costs.append(cost)
            final_epoch_cost = cost
        
        # model.costs.append(final_epoch_cost)
        
        if print_cost:
            print(f"Cost after epoch {itr + 1}: {final_epoch_cost}")


GD_Variants = {
    'BatchGD': BatchGD,
    'StochasticGD': StochasticGD
}

