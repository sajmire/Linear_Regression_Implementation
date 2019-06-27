#!/usr/bin/env python
# coding: utf-8

# In[293]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import math
import itertools


# In[238]:


def read_data(data):
    # Reading .txt file into csv dataframe
    df = pd.read_csv(data, header = None)
    return df


# In[239]:


def visualize_data(x,y):
    # Plotting points as a scatter plot 
    plt.scatter(x, y, label= "stars", color= "green", marker= "*", s=30) 
    plt.xlabel("Population of City in 10,000s") 
    plt.ylabel("Profit in $10,000s") 
    plt.title("Plot of population v/s profit") 
    # Showing legend 
    plt.legend() 
    # Function to show the plot 
    plt.show() 


# In[240]:


def feature_normalization(df,mean,std):
    # Need normalization when many features are there, so that all the features are on similar scale
    norm_df = (df-mean)/std
    return norm_df


# In[241]:


def get_initial_values(df):
    x=[]
    temp_df = df.loc[:, df.columns != df.columns[len(df.columns)-1]] # Delete last column which is class value(y)
    if(len(temp_df.columns)>1):
        temp_df = feature_normalization(temp_df,temp_df.iloc[:,0:2].mean(),temp_df.iloc[:,0:2].std())
        
    # Inserting 1 at index 0, for simplifing the hypothesis equation
    for i in range(len(temp_df)):
        values = list(temp_df.loc[i].values)
        values.insert(0,1)
        x.append(values)
    x = np.array(x)
    # Get seperate colum for class Value (y)
    y = [df.iloc[i][len(df.columns)-1] for i in range(len(df))]
    
    # Initialize theta to vector of 0 having length = number of features
    theta = np.zeros((len(df.columns),1),float)
    return x,y,theta


# In[242]:


def compute_cost(x,y,theta):
    # hypothesis = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*theta)] for X_row in x]
    # Hypothesis = theta(0)*x0 + theta(1)*x1 + ... + theta(n)*xn or (transpose of theta) * x
    hypothesis = [np.dot(x[i].transpose(),theta) for i in range(len(x))]
    loss = [hypothesis[i] - y[i] for i in range(len(x))]
    
    # Cost = 1/2m * sumation((hypothesis/prediction - actual value y)^2) where m is number of examples
    cost = (sum([math.pow(loss[i],2) for i in range(len(x))]))/(2*len(x))
    print(cost)


# In[303]:


def gradient_descent(x,y,theta):
    # In gradient descent we will update the value of theta so that cost will minimize
    # theta = theta - alpha * (1/m * loss * x)
    theta_list = []
    cost_list = []
    iterations = 1500
    alpha = 0.01
    for iteration in range(iterations):  
        hypothesis = [np.dot(x[i].transpose(),theta) for i in range(len(x))]
        theta_list.append(theta)
        loss = [hypothesis[i] - y[i] for i in range(len(x))]
        cost = (sum([math.pow(loss[i],2) for i in range(len(x))]))/(2*len(x))
        cost_list.append(cost)
        gradient = np.dot(x.transpose(),loss)/len(x)
        theta = theta - alpha * gradient
    print("cost:%s and theta:%s" %(cost,theta))
    return theta


# In[253]:


def prediction_data(data,mean,std):
    if(len(data)>1):
        df = pd.DataFrame([data])
        df = feature_normalization(df,mean,std)
        # # df1 = array.values.tolist()
        df = list(itertools.chain(*(df.values.tolist())))
        df.insert(0,1)
    else:
        df = data
        df.insert(0,1)
    return np.array(df)


# In[259]:


def prediction(feature_array,theta):
    predicted_value = np.dot(feature_array.transpose(),theta)
    print("predicted value for features %s:%s" %(feature_array[1:len(feature_array)],predicted_value))
    return predicted_value


# In[304]:


def main():
    data_one_var = "ex1data1.txt"
    df_one_var = read_data(data_one_var) 
    visualize_data(df_one_var[0], df_one_var[1])
    x,y,theta = get_initial_values(df_one_var)
    compute_cost(x,y,theta)
    final_theta = gradient_descent(x,y,theta)
    data = [3.5]
    data = prediction_data(data,df_one_var.iloc[:,0:2].mean(),df_one_var.iloc[:,0:2].std())
    prediction(data,final_theta)
    
    data_many_var = "ex1data2.txt"
    df_many_var = read_data(data_many_var) 
    x,y,theta = get_initial_values(df_many_var)
    compute_cost(x,y,theta)
    final_theta = gradient_descent(x,y,theta)
    data = [1600,3]
    data = prediction_data(data,df_many_var.iloc[:,0:2].mean(),df_many_var.iloc[:,0:2].std())
    prediction(data,final_theta)

if __name__ == '__main__':
    main()

