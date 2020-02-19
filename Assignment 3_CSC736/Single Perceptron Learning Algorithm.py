# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:06:50 2020

@author: rit1115
"""

import random 
import matplotlib.pyplot as plt 
import numpy as np



def line(x):
    ''' an Arbitrary equation of line. takes X points and return Y points
    '''
    #y = x-30
    val = x[1]-x[0]+30

    return val > 0 and 1 or -1
def draw_line(x):
    y = x-30
    return y
def activate(sum):
    return sum>0 and 1 or -1
def main():
    plt.ion()
    
    number_of_points = 20
    # Generating 20 random values for X
    X = [[random.randint(0,1000),random.randint(0,1000)] for i in range(number_of_points) ]

    point_class = [line(x) for x in X]


    X = np.array(X)
    
    # initializing bias and with
    bias = random.uniform(-1,1.0)
    weight1 = random.uniform(-1,1.0)
    weight2 = random.uniform(-1,1.0)
    n = 0.000001
    epoch=0
    while(True):
        epoch+=1
        
        # Training linear model
        miss_Classified =0
        for index in range(number_of_points):
            sum = bias + weight1 * X[index][0]+weight2 * X[index][1] 
            output = activate(sum)
            error = point_class[index]- output
            if output != point_class[index]:
                weight1 += n * error * X[index][0]
                weight2 += n * error * X[index][1]
                bias += n * error*bias
                miss_Classified+=1

    
        y = draw_line(X)
        
        for i in range(len(X)):
            if(point_class[i] == -1):
                plt.scatter(X[i][0],X[i][1],color='r',facecolors='none', edgecolors='r')
            else:
                plt.scatter(X[i][0],X[i][1],facecolors='b', edgecolors='b')
        
        xx = np.linspace(0,1000,100)
        yy = abs((-weight1*xx - bias)/float(weight2)) 
        plt.plot(X,y,'-g')
        plt.plot(xx,yy,'-y')
        plt.draw()    
        plt.pause(0.1) # change to make fast or slow animation
        plt.clf()
        print("Epoch %3d Misclassification =%d"%(epoch+1,miss_Classified))
        if (miss_Classified==0):
            break;
main()
        