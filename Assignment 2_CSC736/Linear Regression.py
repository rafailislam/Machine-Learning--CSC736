
"""
Created on Thu Feb  6 16:34:41 2020

@author: rit1115
"""
import random 
import matplotlib.pyplot as plt 
import numpy as np



def line(x):
    ''' an Arbitrary equation of line. takes X points and return Y points
    '''
    y = 2*x+3
    return y
def main():
    plt.ion()
    
    number_of_points = 20
    # Generating 20 random values for X
    X = [random.randint(0,500) for i in range(number_of_points) ]
    Y = []
    yy = []
    
    # calculating y points from the line equation
    for i in X:
        y = line(i)
        yy.append(y)
        y = y + y * 0.1*random.choice([1,-1])
        
        Y.append(y)

    X = np.array(X)
    
    # initializing bias and with
    bias = random.uniform(0,1.0)
    weight = random.uniform(0,1.0)
    n = 0.000001
    
    for epoch in range(500):
        y_predicted=[]
        
        # Training linear model
        for point in range(number_of_points):
            y_val = weight*X[point]+bias
            y_predicted.append(y_val)
        
        errB = 0
        errW = 0
        mse = 0
        
        # calculating  accumulated error for bias, weight and MSE
        for i in range(20):
            val = Y[i]-y_predicted[i]
            errB += val
            errW += val*X[i]
            mse = val**2
        
        mse = mse/number_of_points
        errB= errB/number_of_points
        errW = errW/number_of_points
        # updating bias and weight
        bias = bias + n * errB
        weight = weight + n * errW
        
        y = line(X)
        plt.scatter(X,Y)
        plt.plot(X,y,'-g')
        plt.plot(X,y_predicted,'-y')
        #plt.show()

        plt.draw()    
        plt.pause(.001)
        plt.clf()
        print("Epoch %3d MSE =%4.3f"%(epoch+1,mse))
main()
        