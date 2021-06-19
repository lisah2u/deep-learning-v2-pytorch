# plot lines
# %%
import numpy as np
import matplotlib.pylab as plt

# %%
# TODO: color specific points
x = np.array([0, 0, 1, 1])
y = np.array([0, 1, 1, 0])
plt.plot(x, y, 'o')
plt.plot
# plot a best fit line through these points
m, b = np.polyfit(x, y, 1)
print(m, b)

plt.plot(x, m*x + b)

# %%

def graph(formula, x_range):
    x = np.array(x_range)
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)
    plt.show()

def my_formula(x):
    return 3*x+4*x-10

graph(my_formula, range(0, 2))

# %%
# Ax + By = C
# 3x1 + 4x2 - 10 = 0
# 3*x+4*x-10
# X intercept is C/A
# Y intercept is C/B

x = [10/3,0]
y = [0,10/4]
plt.plot(x,y,'-')
plt.show()
# %%
