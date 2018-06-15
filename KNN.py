# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 22:28:05 2018

@author: dell1
"""


from sklearn.datasets import load_iris

data = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['data'],data['target'], random_state = 0)

import time
import numpy as np
import math
import matplotlib.pyplot as mp

accuracy = []
time_arr = []

class KNN(object):
    def __init__(self):
        pass
    
    
    def train(self, X, y):
        """
        X = X_train
        y = y_train
        """
        self.X_train = X_train
        self.y_train = y_train
       
    def predict(self, X_test, k): 
        """
        It takes X_test as input, and return an array of integers, which are the 
        class labels of the data corresponding to each row in X_test. 
        Hence, y_project is an array of lables voted by their corresponding 
        k nearest neighbors
        """
        y_predict = []

        for x in range(len(X_test)):
            X_arr = []
            nn =[]
            for i in range(len(X_train)):
                op = math.sqrt(sum((X_test[x] - X_train[i])**2))
                X_arr = np.append(X_arr,[op])
                
            idx = np.argsort(X_arr)
            nn = idx[:k]
            neigh = []
            for x in range(len(nn)):
                given = y_train[nn[x]]
                neigh.append(given)
                
            count_zero = neigh.count(0)
            count_one = neigh.count(1)
            count_two = neigh.count(2)
            max_count = max([count_zero, count_one, count_two])
            #print(max_count)
            if((count_zero == max_count and count_one == max_count) or
               (count_one == max_count and count_two == max_count) or
               (count_two == max_count and count_zero == max_count)):
                
                y_predict.append(-1)
                #print("if", y_predict)
            else:
                if(count_zero == max_count):
                    y_predict.append(0)
                elif(count_one == max_count):
                    y_predict.append(1)
                elif(count_two == max_count):
                    y_predict.append(2)
                #print('else', y_predict)
                
        return y_predict        
            

    def report(self,X_test, y_test, k):
        """
        return the accurancy of the test data. 
        """
        count = 0
        #print(time_stamp)
        time_stamp = time.clock()
        y_predict = test.predict(X_test, k)
        time_done = time.clock()
        
        time_taken = time_done - time_stamp
        time_arr.append(time_taken)
        #print(time_arr)
        for acc in range(len(y_test)):
            if(y_predict[acc] == y_test[acc]):
                count+=1
        accuracy = (count/38)*100
        print('Please wait. Plotting the accuracy..')
        return accuracy


def k_validate():
    """
    plot the accuracy against k from 1 to a certain number so that one could pick the best k
    """
    k = range(1,113)
    mp.plot(k, accuracy)
    mp.xlabel('k value')
    mp.ylabel('accuracy')
    mp.show()

test = KNN()
for k in range(len(X_train)):   
    accuracy.append(test.report(X_test, y_test,k+1))

print('The best accuracy is ',max(accuracy),'against k = ', accuracy.index(max(accuracy))+1)
t = min(time_arr)
k_min = time_arr.index(t)+1
print('The best time is ',t,'sec for k = ',k_min)

k_validate()
