# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:31:38 2020

@author: rit1115
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

output_vector=[[0.9,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.9,0.1],[0.1,0.1,0.1,0.9]]
# learning rate
n = 0.01
a = 1
wji = []
wkj =[]
delta_j=[]
delta_k=[]
y_k=[]
bias_j=[]
bias_k=[]


def read_traing_data():
'''This function reads training data set from input file and normalized it.
'''

    train=[]
    train_output=[]
    with open('optdigits-3.tra') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            rows=[]
            for i in range(64):
                rows.append( (int(row[i]))/16) # dividing values by 16 to convert it into 0-1 range
            #print(len(rows))
            output = int(row[64])
            #print(output)
            train_output.append(output_vector[output])
            train.append(rows)
    return train,train_output

def read_testing_data():
'''This function reads testing data set from input file and normalized it.
'''

    test=[]
    test_output=[]
    with open('optdigits-3.tes') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            #print(len(row))
            rows=[]
            
            for i in range(64):
                rows.append( (int(row[i]))/16) # dividing values by 16 to convert it into 0-1 range
            #print(len(rows))
            output = int(row[64])
            #print(output)
            test_output.append(output_vector[output])
            test.append(rows)
    return test, test_output


def initialization(no_hidden_node):
    ''' this function takes the number of nodes in hidden layer and return weight both for input-hidded and hidden-output layer.
    it also returns bias and delta
    '''
    global wji 
    global wkj
    global delta_j 
    global delta_k 
    global y_k 
    global bias_j 
    global bias_k 
    
    for _ in range(64*no_hidden_node):
        wji.append(round(np.random.uniform(-1,1),3))
   
    wji= np.reshape(wji,(64,no_hidden_node)).tolist()
    
    wkj=[round(np.random.uniform(-1,1),3) for _ in range(no_hidden_node*4)] 
    wkj= np.reshape(wkj,(no_hidden_node,4)).tolist()
   
    
    delta_j = [0.0]*no_hidden_node
    bias_j  = [round(np.random.uniform(-1,1),3) for _ in range(no_hidden_node)] 
    delta_k = [0.0]*4
    bias_k = [round(np.random.uniform(-1,1),3) for _ in range(4)] 
    y_k = [0.0]*4
    
    
    return wji,wkj,delta_j,delta_k,y_k,bias_j,bias_k


def input_hidden_layer(inputs):
    ''' this function calculate the output of input-hidden layer for each data set.
    '''
    y_j = []
    for j in range(5):
        summ = 0
        for i in range(64):
            summ += inputs[i]*wji[i][j]
        summ = summ + bias_j[j]   
        y_j.append(activation(summ))     
    #print(len(y_j))
    
    return y_j

def hidden_output_layer(y_j):
    ''' this function calculate the output of hidden-output layer for each data set.
    '''
    y_k=[]
    
    for i in range(4):
        summ = 0
        for j in range(5):
            summ += y_j[j]*wkj[j][i]
        summ = summ + bias_k[i]    
        y_k.append(activation(summ))

    return y_k
    
def feed_forward(inputs):
'''this function takes input and feed_forward inputs to determine outputs.
'''
    y_j = input_hidden_layer(inputs)
    y_k = hidden_output_layer(y_j)
    return y_j,y_k
    
def back_propagate(inputs,desired_outputs, y_j, y_k):
    ''' this function calculate error of each layer and adjust weights according to the errors the generated. 
    '''
    # error in output layer
    for i in range(len(desired_outputs)):
        delta_k[i] = (a * y_k[i] * (1 - y_k[i]) * (desired_outputs[i] - y_k[i]))
    
    # error in hidden layer
    for i in range(len(y_j)):
        sum=0
        for j in range(len(delta_k)):
            sum += delta_k[j] * wkj[i][j]
        delta_j[i] = a* y_j[i]*(1-y_j[i])*sum
    
    # update hidden-output layer's weights
    for j in range(5):
        for k in range(4):
            wkj[j][k] = wkj[j][k] + n*delta_k[k]*y_j[j]
    
    
    # update bias of output layer
    for k in range(4):
        bias_k[k] += n*bias_k[k]*delta_k[k]
        
    # update input-hidden layer's weights
    for i in range(64):
        for j in range(5):
            wji[i][j] = wji[i][j]+ n * delta_j[j]*inputs[i]
    
    # update bias of output layer
    for j in range(5):
        bias_j[j] += n*bias_j[j]*delta_j[j]
        
        
    
def fcn_train(train_input_set,desired_outputs):
    # train single data set at a time
    # stochastic gradient descent
    for i in range(len(train_input_set)):
        y_j,y_k = feed_forward(train_input_set[i])
        back_propagate(train_input_set[i], desired_outputs[i], y_j, y_k)
         
def mse_calculation(inputs,actual_output):
'''this function calculate outputs for given set of data and compare outputs with actual_output with predicted outputs
and calculate Mean Square Error
'''
    predicted_output=[]
    for i in range(len(inputs)):
        y_j,y_k = feed_forward(inputs[i])
        predicted_output.append(y_k)
    
    sum_square= 0
    for i in range(len(predicted_output)):
        for j in range(4):
            sum_square += (actual_output[i][j] - predicted_output[i][j])**2
    MSE = sum_square / 2
    return MSE    
    

# Sigmoid activation function
def activation(z):
    return 1 / (1 + np.exp(-z))


def correctness(testing,disired_output):
''' This function takes testing data set and pedict output of the testing data set and compute 
the correctness of the predicted output with desired ouputs.
'''
    predicted_output=[]
    for i in range(len(testing)):
        y_j,y_k = feed_forward(testing[i])
        predicted_output.append(y_k)
    
    correct = 0
    # this loop iterates through all the output and compare predicted outputs with desired outputs.
    for i in range(len(predicted_output)):
        if( disired_output[i].index(max(disired_output[i])) == predicted_output[i].index(max(predicted_output[i]))):
            correct += 1
            
    print("no of correct", correct)
    # convert correctness into percentage
    percentage = correct/len(predicted_output) *100 
    
    return percentage


def main():
    
   
    global n
    # no of nodes in hidden layer
    no_hiddenlayer_node = 5
    
    # Reading training data from optdigits-3.tra and normalized it
    train, train_output = read_traing_data()
    
    # Reading testing data from optdigits-3.tra and normalized it
    test, test_output = read_testing_data()
    
    # spliting traning data into 80% for training and 20% for validation
    
    train_input_set,validation_input_set, tarin_output_set,  validation_output_set = train_test_split(train, train_output, test_size=0.20, random_state=32)
    
    initialization(no_hiddenlayer_node)
    
    mse_traing = []
    mse_validation = []
    x_axis=[]
    epoch = 1
    temp = 9999999
    while(epoch<20000):
        plt.clf()
        
        fcn_train(train_input_set,tarin_output_set)
        if(epoch%10==0):
        
            mse_t =mse_calculation(train_input_set,tarin_output_set)
            mse_v = mse_calculation(validation_input_set,validation_output_set)
            
            mse_traing.append( mse_t) 
            mse_validation.append(mse_v)
            if(temp<mse_v):
                break
            temp = mse_v
            
            x_axis.append(epoch)
            plt.plot(x_axis, mse_traing, label = 'Training Data Set' )
            plt.plot(x_axis, mse_validation, label = 'Validation Data Set' )
            plt.xlabel( "Number of epochs" )
            plt.ylabel( "Mean Square Error" )
            plt.title( "Mean Square Error vs Epochs" )
            plt.draw( )
            plt.pause( 0.00000001 )
        #end of if
        
        epoch += 1
        
        if epoch % 800 ==0:
            n = n + n * 0.8
            print("epoch: ", epoch)
    #end of while
    
    df = pd.DataFrame({
        "Traing": mse_traing,
        "Validation ": mse_validation 
    })
    ax = df.plot()
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    # calculate accuracy with testing data set
    correct_classification = correctness(test,test_output) # even same train set doesnot work well
    print("Correct Cllasificaoin : %.2f %%"%(correct_classification))
    input ( "press [enter] to exit" )

    
main()