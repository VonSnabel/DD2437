import numpy as np


#Needs work
def deltaRule(X, T, lr, epochs):

    samples, features = X.shape

    W = np.random.randn(features)
    
    error = W@X-T

    delta_W = lr*(error)*X.T

    return W

def generateData():
    n=100

    mA = np.array([1.0, 0.5])
    sigmaA = 0.5

    mB = np.array([-1.0, 0.0])
    sigmaB = 0.5

    classA = np.zeros((2, n))
    classA[0,:] = np.random.randn(n) * sigmaA *mA[0]
    classA[1,:] = np.random.randn(1, n) *sigmaA*mA[1]

    classB = np.zeros((2, n))
    classB[0,:] = np.random.randn(n) * sigmaB *mB[0]
    classB[1,:] = np.random.randn(n) *sigmaB*mB[1]



def main():
    

    return


