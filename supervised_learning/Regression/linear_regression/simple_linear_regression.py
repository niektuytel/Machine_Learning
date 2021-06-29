import csv
import numpy as np
import matplotlib.pyplot as plt

# value = (value - lowest value) <-- set lowest value to 0
# return = 0.% of (value / max value)
def	normalizeElem(list_values, elem):
	return ((elem - min(list_values)) / (max(list_values) - min(list_values)))

# value = (highest value - lowest value) <-- set lowest value to 0
# return = (0.% * value) + lowest value 
def	denormalizeElem(list_values, elem):
	return ((elem * (max(list_values) - min(list_values))) + min(list_values))

# info: https://github.com/niektuytel/ML_Algorithms/gradient_descent#scaling
def normalizeData(X, Y):
    new_X = []
    new_Y = []

    for value in X:
        new_X.append(
            normalizeElem(X, value)
            # [normalizeElem(X, value)]
        )

    for value in Y:
        new_Y.append(
            normalizeElem(Y, value)
            # [normalizeElem(Y, value)]
        )

    return new_X, new_Y# np.array(new_X), np.array(new_Y)

def getData(fileName):
    X = []
    Y = []

    with open(fileName, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',' )
        for row in csvReader:
            X.append(eval(row[0]))
            Y.append(eval(row[1]))

    return X, Y

def setData(fileName, X, Y):
    # write data to given filename
    with open(fileName, 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csvWriter.writerow([X, Y])

# info https://github.com/niektuytel/ML_Algorithms/gradient_descent/tree/main/cost_functions  
def	costFunction(theta, X, Y):
    cost = 0.0

    for i in range(len(X)):
        prediction = (theta[1] * X[i] + theta[0])        
        cost = cost + (prediction - Y[i]) ** 2

    return (cost / len(X))

def	boldDriver(cost, cost_history, theta, batch_theta, learning_rate, length):
    new_learning_rate = learning_rate
    if len(cost_history) > 1:
        if cost > cost_history[-1]:
            theta = [
                theta[0] + batch_theta[0] / length * learning_rate,
                theta[1] + batch_theta[1] / length * learning_rate,
            ]
            new_learning_rate *=  0.5
        else:
            new_learning_rate *= 1.05

    return theta, new_learning_rate

def earlyExit(cost_history):
    len_check = 8
    if len(cost_history) > len_check:
        mean = sum( cost_history[-(len_check):] ) / len_check
        last = cost_history[-1]
        if round(mean, 9) == round(last, 9): 
            return True
    return False

def displayPlot(X, Y, theta, cost_history, theta_history):
    # final ouput line
    lineX = [float(min(X)), float(max(X))]
    lineY = []
    for elem in lineX:
        elem = theta[1] * normalizeElem(X, elem) + theta[0]
        lineY.append(denormalizeElem(Y, elem))

    # display the linear regression line result
    plt.figure(1)
    plt.plot(X, Y, 'bo', lineX, lineY, 'r-')
    # plt.plot(theta_history, 'b.')
    plt.xlabel('x')
    plt.ylabel('y')

    # display cost on the interaction index
    plt.figure(2)
    plt.plot(cost_history, 'r.')
    plt.xlabel('iterations')
    plt.ylabel('cost')

    # display
    plt.show()

# info: https://github.com/niektuytel/ML_Algorithms/gradient_descent
def	gradientDescent(theta, cost_history, X, Y, learning_rate, iterations):
    batch_theta = [0, 0]
    for coordinateX, coordinateY in zip(X, Y):
        total = (theta[1] * coordinateX + theta[0]) - coordinateY
        batch_theta = [
            batch_theta[0] + total, 
            batch_theta[1] + (total * coordinateX)
        ]
        
    theta = [
        theta[0] - batch_theta[0] / len(X) * learning_rate,
        theta[1] - batch_theta[1] / len(Y) * learning_rate
    ]

    # position slope
    cost = costFunction(theta, X, Y)

    # rotation slope
    theta, learning_rate = boldDriver(
        cost,
        cost_history, 
        theta, 
        batch_theta, 
        learning_rate, iterations
    )

    return theta, cost, learning_rate

# info: https://github.com/niektuytel/ML_Regressions/linear_regression
def linearRegression(X, Y, learning_rate = 0.1, iterations = 1000):
    theta = [0.0, 0.0]
    theta_history = [0.0, 0.0]
    cost_history = []

    for iteration in range(iterations):
        theta, cost, learning_rate = gradientDescent(theta, cost_history, X, Y, learning_rate, iterations)
        theta_history.append(theta)
        cost_history.append(cost)

        if iteration % 10 == 0:
            print("epoch {} - cost: {:.8}".format(iteration, cost))

        # if cost not change -> exit
        if earlyExit(cost_history):
            print("(Early Exit)")
            break

    print("iteration {} - cost: {:.8} - theta: {}".format(iteration, cost, theta))
    return theta, cost_history, theta_history

if __name__ == "__main__":
    X, Y = getData("../_DATA/data_1.csv")
    normal_X, normal_Y = normalizeData(X, Y)
    
    theta, cost_history, theta_history = linearRegression(normal_X, normal_Y)
    setData("../_DATA/theta.csv", theta[0], theta[1])
    displayPlot(X, Y, theta, cost_history, theta_history)