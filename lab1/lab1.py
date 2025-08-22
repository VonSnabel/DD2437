import numpy as np

def unitStep(x):
    return np.where(x > 0, 1, 0)


class Perceptron:

    def __init__(self, learningRate=0.2, n=100):
        self.lr = learningRate
        self.n = n
        self.activation = unitStep
        self.weights = None
        self.bias = None

    def fit(self, X, t):
        nSamples, nFeatures = X.shape

        self.weights = np.zeros(nFeatures)
        self.bias = 0

        t_ = np.where(t>0, 1, 0)

        for _ in range(self.n):
            for i, xi in enumerate(X):
                linOut = np.dot(xi, self.weights)+self.bias
                tPred = self.activation(linOut)

                update = self.lr * (t_[i]-tPred)
                self.weights += update*xi
                self.bias += update

    def predict(self, X):
        linOut = X@self.weights+self.bias
        tPred = self.activation(linOut)
        return tPred
    
if __name__ == "__main__":

        import matplotlib.pyplot as plt


        def accuracy(tTrue, tPred):
             return (np.sum(tPred==tTrue)/len(tTrue))
        
        
        
        