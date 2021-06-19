# perceptron trick
# %%
# where the line is described by 3x1+ 4x2 - 10 = 0, if the learning rate was set to 0.1, how many times would you have to apply the perceptron trick to move the line to a position where the blue point, at (1, 1), is correctly classified?

# %%
# Starting line

import matplotlib.pyplot as plt
import numpy as np
def draw_line(a, b, c, color='black', linewidth=2.0, linestyle='solid', start=0, stop=3):
    # Plotting the line ax + by + c = 0
    x = np.linspace(start, stop)
    plt.plot(x, -c/b - a*x/b, linestyle=linestyle,
                color=color, linewidth=linewidth)

def graph():
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(1,1,'o')
    draw_line(3,4,-10)
    plt.title("3x + 4y = 10")
    plt.show()

graph()
# %%
""" 
• Use the point values to modify the parameter of the line(+ 1 for bias - y).
• Multiple the learning rate times the point values.
• Add / subtract that value to the parameter.
• If the point is negative in the positive area, subtract each value
• If the point is positive in a negative area, add each value
• See the line move closer to the point.
• Add all three values to see if the are >= or < 0
• If the point is negative and in the positive area, it should be < 0
• If the point is positive and in the negative area, it should be >= 0 

Pseudocode:

features = [1,1] # for more points, make an np.array of lists
We know already that [1,1] is positive in a negative area. 
It has a negative score and we are trying to get to positive.
labels = [1] # np.array for more than one label
weights = 3,4
bias = -10
learning_rate = .1
global counter = 0

If the number is below 0, do the perceptron trick again
step_function(num)

    if num < 0
        do the perceptron trick again
        add to a global counter
    elif num >=0
        return counter

Score the output
score(weights, bias)
    total += weights + bias
    return total

We already know this is a +point in a -area.
If we had to determine this, we'd check a label to 
see if it's +/- and also if the prediction was +/-
When they don't match, you add or subtract accordingly

perceptron_trick
    for each weight
        weight +=feature * learning_rate
    bias += learning_rate
    score(weights, bias)
    step_function(total)

"""
# %%
feature = [1, 1]  # for more points, make an np.array of lists
# label = [1]  # np.array for more than one label. Only is we are 
weights = [3, 4]
bias = -10
learning_rate = .1
counter = 0

# If the number is below 0, do the perceptron trick again
def step(total):
    if total < 0:
        perceptron_trick(weights, bias)
    elif total >= 0:
       return

def score(weights, bias):
    total = 0
    total += weights[0] + weights[1] + bias
    return total

def perceptron_trick(weights, bias):
    for weight in range(len(weights)):
        weight += feature[weight] * learning_rate
    bias += learning_rate
    total = score(weights, bias)
    step(total)
    print(counter)

perceptron_trick(weights, bias)

# %%
