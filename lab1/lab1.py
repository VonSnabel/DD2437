import numpy as np

def unitStep(x):
    return np.where(x > 0, 1, 0)


class Perceptron:

    def __init__(self, learningRate=0.2, epochs=100):
        self.lr = learningRate
        self.epochs = n
        self.activation = unitStep
        self.weights = None
        self.bias = None

    def fit(self, X, t):
        nSamples, nFeatures = X.shape

        self.weights = np.zeros(nFeatures)
        self.bias = 0

        t_ = np.where(t>0, 1, 0)

        for _ in range(self.epochs):
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
        
        def generateInput():
             n = 100
             
             mA = np.array([1.0, 0.5])
             sigmaA = 0.5
             classA = np.random.randn(2, n) *sigmaA +mA[:, np.newaxis]
             
             mB = np.array([-1.0, -0.0])
             sigmaB = 7
             classB = np.random.randn(2, n) *sigmaB +mB[:, np.newaxis]

             return [classA, classB]
        