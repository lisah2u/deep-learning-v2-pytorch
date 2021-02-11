# experiments.py
# answer - https://classroom.udacity.com/nanodegrees/nd101/parts/94643112-2cab-46f8-a5be-1b6e4fa7a211/modules/4d61ad35-d227-460e-8be6-16407f273e7d/lessons/00912248-2445-4713-ad9c-76b8536e1959/concepts/9e1364a8-e8b4-4eac-be12-4d44a139f721#
#%%
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
#%%
# Some data 
# ingest csv 
df = pd.read_csv('data.csv',
                 delimiter=',', header=None, names=['x', 'y','color'])
for index, row in df.iterrows():
    if row.color == 1.0:
        plt.scatter(row.x, row.y, c="red")
    if row.color == 0.0:
        plt.scatter(row.x, row.y, c="blue")

plt.xlabel('x')
plt.ylabel('y')
plt.title('scatterplot')
#plt.legend()
plt.show()
#%%
x = np.arange(-5, 5, .1)
def softmax(x):
    f = 1 / (1 + np.exp(-x))
    return f
#%% 
# Plot softmax
f = softmax(x)
plt.plot(x,f)
plt.style.use('seaborn-whitegrid')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
# %%
# Plot a line
#  3x1 + 4x2 - 10 = 0
x = np.linspace(3,4,5)
y = 3*x + 4*x + 10
#x = np.linspace(-5, 5, 100)
#y = 2*x+1
plt.plot(x, y, '-r', label='y=2x+1')
plt.title('Graph of y=2x+1')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(-5, 5, 100)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(x, 2*x+1, '-r', label='y=2x+1')
plt.plot(x, 2*x-1, '-.g', label='y=2x-1')
plt.plot(x, 2*x+3, ':b', label='y=2x+3')
plt.plot(x, 2*x-3, '--m', label='y=2x-3')
plt.legend(loc='upper left')
plt.show()

# %%
