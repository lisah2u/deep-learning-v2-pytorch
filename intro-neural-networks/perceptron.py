#%%
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

#%%
def stepFunction(t):
    if t >= 0:
        return 1
    return 0
#%%
def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])
#%%
# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.


def perceptronStep(X, y, W, b, learn_rate=0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
#%%
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 1):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

# %%

def draw_line(a, b, c, color='black', linewidth=2.0, linestyle='solid', starting=0, ending=3):
    # Plotting the line ax + by + c = 0
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, -c/b - a*x/b, linestyle=linestyle,
             color=color, linewidth=linewidth)
# %%
df = pd.read_csv('data.csv',
                 delimiter=',', header=None, names=['x', 'y', 'color'])
for index, row in df.iterrows():
    if row.color == 1.0:
        plt.scatter(row.x, row.y, c="red")
    if row.color == 0.0:
        plt.scatter(row.x, row.y, c="blue")

plt.xlabel('x')
plt.ylabel('y')
plt.title('scatterplot')
#plt.legend()

# %% 
# X = pd.DataFrame([[1, 0], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [3, 2], [2, 3]])
# y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])”

# Remove column from DataFrame as Series
# Call trainPerceptronAlgorithm(X=X,y=y)

# plot boundary lines

plt.show()
