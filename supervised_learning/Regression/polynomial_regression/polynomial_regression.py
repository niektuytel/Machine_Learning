import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# quit loop when no changes
def earlyExit(cost_history):
    check = 8
    if len(cost_history) > check:
        mean = np.mean(cost_history[-check:])
        if mean == cost_history[-1]:
            return True
    return False

# ./images/hypothesis.png
def hypothesis(X, theta):
    return np.sum((theta * X), axis=1)

# ./images/cost.png
def cost(X, y, theta):
    h = hypothesis(X, theta)
    return sum((h-y) ** 2)  / (2 * m)

# ./images/simple_linear_formules.png
def gradient_descent(X, y, theta, learning_rate, iterations):
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(theta)
        # print('last 4 predictions : ', predictions[:4])

        errors = (predictions - y)
        # print('last 4 errors      : ', errors[:4])

        theta = theta - (learning_rate / m) * X.T.dot(errors);
        # print('theta              : ', theta)
        
        cost_history.append(cost(X, y, theta))
        # print('cost               : ', cost)
        # print("\n===\n")

        if earlyExit(cost_history):
            print("(Early Exit) on iteration : ", i)
            break

    return cost_history, theta


# get data from file
df = pd.read_csv("../_DATA/data_5.csv")
df.head()

# Add the bias column for theta 0
df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)
df.head()

# remove data
df = df.drop(columns='Position')

y = df['Salary']
X = df.drop(columns = 'Salary')
X.head()

X['Level1'] = X['Level']**2
X['Level2'] = X['Level']**3
X.head()

m = len(X)
X = X/X.max()
learning_rate = 0.1
iterations = 1000

theta = np.zeros(len(X.columns))
J, theta = gradient_descent(X, y, theta, learning_rate, iterations)
y_hat = hypothesis(X, theta)


# final 
plt.figure()
plt.scatter(x=X['Level'],y= y)           
plt.scatter(x=X['Level'], y=y_hat)
plt.show()

plt.figure()
plt.scatter(x=list(range(0, iterations)), y=J)
plt.show()