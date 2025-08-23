import numpy as np

def unitStep(x):
    return np.where(x > 0, 1, -1)


class Perceptron:

    def __init__(self, learningRate=0.2, epochs=100):
        self.lr = learningRate
        self.epochs = epochs
        self.activation = unitStep
        self.weights = None
        self.bias = None
        self.errorHistory = []

    def fit(self, X, t):
        nSamples, nFeatures = X.shape

        self.weights = np.zeros(nFeatures)
        self.bias = 0

        t_ = np.where(t>0, 1, -1)
        self.errorHistory = []

        for epoch in range(self.epochs):
            errors = 0
            for i, xi in enumerate(X):

                linOut = xi @ self.weights + self.bias
                tPred = self.activation(linOut)

                if t_[i] != tPred: 
                    update = self.lr * (t_[i]-tPred)
                    self.weights += update*xi
                    self.bias += update
                    errors += 1
            self.errorHistory.append(errors)
            if errors == 0:
                 break
            self.plotBoundry(X, t, epoch)

    def predict(self, X):
        linOut = X @ self.weights + self.bias
        tPred = self.activation(linOut)
        return tPred
    
    def plotBoundry(self, X, t, epoch):
         plt.scatter(X[t >= 0.5, 0], X[t >= 0.5, 1], c='r', marker='x', label='Class A')
         plt.scatter(X[t < 0.5, 0], X[t < 0.5, 1], c='b', marker='o', label='Class B')

         x = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)

         y = -(self.weights[0]*x+self.bias)/self.weights[1]

         plt.plot(x, y, color='black')
         plt.title('Decision Boundry - Epoch %d' %(epoch+1))

         plt.legend()
         plt.grid()
         plt.show(block=False)
         plt.pause(0.2)
         plt.clf()
    
    def plotTraining(self, epoch):
         plt.plot(range(len(self.errorHistory)), self.errorHistory)
         plt.title('Learning Curve')
         plt.grid()
         plt.show()
    
if __name__ == "__main__":

        import matplotlib.pyplot as plt
        
        def generateInput():
             np.random.seed(1337)
             n = 100
             
             mA = np.array([1.0, 0.5])
             sigmaA = 2
             classA = np.random.randn(2, n) *sigmaA +mA[:, np.newaxis]
             
             mB = np.array([-5.0, -5.0])
             sigmaB = 1.0
             classB = np.random.randn(2, n) *sigmaB +mB[:, np.newaxis]

             return [classA, classB]
        
        def plotData(classA, classB):
             plt.scatter(classA[0, :], classA[1, :], color='b', marker='o', label='Class A')
             plt.scatter(classB[0, :], classB[1, :], color='r', marker='x', label='Class B')
             plt.xlabel('X1')
             plt.ylabel('X2')
             plt.legend()
             plt.grid()
             plt.show()
             plt.close()

        classA, classB = generateInput()

        X = np.concatenate((classA, classB), axis=1).T
        t = np.concatenate((np.zeros(classA.shape[1]), np.ones(classB.shape[1])))

        p = Perceptron(learningRate=0.1, epochs=100)
        p.fit(X, t)

        plotData(classA, classB)
        p.plotTraining()
    




        