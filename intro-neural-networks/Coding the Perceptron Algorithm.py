# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: udl
#     language: python
#     name: udl
# ---

# %% [markdown]
# # Sentiment analysis
#
# ### Some plotting functions

# %%
# Importing packages
from matplotlib import pyplot as plt
import numpy as np
import random


# %%
# Some functions to plot our points and draw the lines
def plot_points(features, labels):
    X = np.array(features)
    y = np.array(labels)
    spam = X[np.argwhere(y==1)]
    ham = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in spam],
                [s[0][1] for s in spam],
                s = 25,
                color = 'cyan',
                edgecolor = 'k',
                marker = '^')
    plt.scatter([s[0][0] for s in ham],
                [s[0][1] for s in ham],
                s = 25,
                color = 'red',
                edgecolor = 'k',
                marker = 's')
    plt.xlabel('aack')
    plt.ylabel('beep')
    plt.legend(['happy','sad'])
def draw_line(a,b,c, color='black', linewidth=2.0, linestyle='solid', starting=0, ending=3):
    # Plotting the line ax + by + c = 0
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, -c/b - a*x/b, linestyle=linestyle, color=color, linewidth=linewidth)


# %%
import pandas as pd
X = pd.DataFrame([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
y = pd.Series([0,0,0,0,1,1,1,1])

# %%
# Plotting the points
plot_points(X, y)

# Uncomment the following line to see a good line fit for this data.
#draw_line(1,1,-3.5)

# %% [markdown]
# ### The Perceptron Algorithm

# %%
def score(weights, bias, features):
    return features.dot(weights) + bias

def prediction(weights, bias, features):
    return int(score(weights, bias, features) >= 0)

def error(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    if pred == label:
        return 0
    else:
        return np.abs(score(weights, bias, features))

def total_error(weights, bias, X, y):
    total_error = 0
    for i in range(len(X)):
        total_error += error(weights, bias, X.loc[i], y[i])
    return total_error


# %%
weights = [1,1]
bias = -3.5
features = X
labels = y
for i in range(len(features)):
    print(prediction(weights, bias, features.loc[i]), error(weights, bias, features.loc[i], labels[i]))


# %%
def perceptron_trick(weights, bias, features, label, learning_rate = 0.01):
    pred = prediction(weights, bias, features)
    if pred == label:
        return weights, bias
    else:
        if label==1 and pred==0:
            for i in range(len(weights)):
                weights[i] += features[i]*learning_rate
            bias += learning_rate
        elif label==0 and pred==1:
            for i in range(len(weights)):
                weights[i] -= features[i]*learning_rate
            bias -= learning_rate
    return weights, bias

def perceptron_trick_clever(weights, bias, features, label, learning_rate = 0.01):
    pred = prediction(weights, bias, features)
    for i in range(len(weights)):
        weights[i] += (label-pred)*features[i]*learning_rate
        bias += (label-pred)*learning_rate
    return weights, bias


# %%
perceptron_trick_clever(weights, bias, features.loc[6], 0)

# %%
random.seed(0)
def perceptron_algorithm(X, y, learning_rate = 0.01, epochs = 200):
    weights = [1.0 for i in range(len(X.loc[0]))]
    bias = 0.0
    errors = []
    for i in range(epochs):
        # Uncomment the following line to draw all the intermediate classifiers
        draw_line(weights[0], weights[1], bias, color='grey', linewidth=1.0, linestyle='dotted')
        errors.append(total_error(weights, bias, X, y))
        j = random.randint(0, len(features)-1)
        weights, bias = perceptron_trick(weights, bias, X.loc[j], y[j])
    draw_line(weights[0], weights[1], bias)
    plot_points(X, y)
    plt.show()
    plt.scatter(range(epochs), errors)
    return weights, bias


# %%
perceptron_algorithm(X, y)

# %%
