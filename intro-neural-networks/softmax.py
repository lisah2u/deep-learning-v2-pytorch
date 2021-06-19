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

