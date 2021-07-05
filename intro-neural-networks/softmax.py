# %%
import numpy as np
# standard exponential function to each element 
# and normalizes these values by dividing by the sum 
# of all these exponentials. This normalization ensures
# that the sum of the components of the output vector 
# is 1.


# Write a function that takes as input 
# a list of numbers, and returns the list
# of values given by the softmax function.

# Returns a list
""" def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i/sumExpL)
    return result """

# Alternatively...
""" def softmax(L):
    exp = np.exp(L)
    return np.divide(exp, exp.sum()) """

# Returns an array
def softmax(L):
    exponentials = np.exp(L)
    sum_exponentials = sum(exponentials)
    result = exponentials/sum_exponentials
    return result

softmax([1,2,3])

# [0.09003057317038046, 0.24472847105479767, 0.6652409557748219]

# %%
# Quiz 
# Based on the above video, let's define the combination 
# of two new perceptrons as w1*0.4 + w2*0.6 + b. 
# Which of the following values for the weights and the 
# bias would result in the final probability of the point 
# to be 0.88?

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

inputs = [.4,.6]
w1 = [2,6]
b1 = -2
w2 = [3,5]
b2 = -2.2
w3 = [5,4]
b3 = -3

print(output_formula(inputs,w2,b2))

# %%
