import matplotlib.pyplot as plt
import numpy as np
import math


def sigmoid(z):
    if isinstance(z, np.matrixlib.defmatrix.matrix):
        for index, x in np.ndenumerate(z):
            z[index[0], index[1]] = sigmoid(x)
        return z
    else:
        return 1/(1+math.exp(-z))


def cost(X, T, y):
    return (-1/len(y))*(np.transpose(y)*np.log(sigmoid(X*T)) + np.transpose(1-y)*np.log(1 - sigmoid(X*T)))


def gradient_descent(X, T, y):
    alpha = 1
    # print(cost(X, T, y))
    m = len(y)
    for x in range(1000):
        T = T - (alpha/m)*(np.transpose(X)*(sigmoid(X*T) - y))
        # print(T)
        # print(cost(X, T, y))
    return T

data = open('ex2data1.txt').read().splitlines()

test1Scores = []
test2Scores = []
y = []

for x in range(len(data)):
    data[x] = data[x].split(',')
    test1Scores.append(float(data[x][0]))
    test2Scores.append(float(data[x][1]))
    y.append(float(data[x][2]))

normalized_test1Scores = (test1Scores - np.mean(test1Scores))/np.std(test1Scores)
normalized_test2Scores = (test2Scores - np.mean(test2Scores))/np.std(test2Scores)

x0 = np.ones(len(y))
X = np.vstack((x0, normalized_test1Scores, normalized_test2Scores))
X = np.transpose(np.matrix(X))

y = np.transpose(np.matrix(y))

T = np.zeros(3)
T = np.matrix(T)
T = np.transpose(T)

T = gradient_descent(X, T, y)

print(cost(X, T, y))

x1 = np.asarray(test1Scores)

x2 = (-float(T[0]) - float(T[1])*normalized_test1Scores)/float(T[2])

x2 = x2*np.std(x1)
x2 = x2 + np.mean(x1)

admitted = [[], []]
notAdmitted = [[], []]

for x in range(len(y)):
    if y[x] == 0:
        notAdmitted[0].append(test1Scores[x])
        notAdmitted[1].append(test2Scores[x])
    else:
        admitted[0].append(test1Scores[x])
        admitted[1].append(test2Scores[x])

plt.axis([30, 100, 30, 100])
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.plot(admitted[0], admitted[1], 'k+', markeredgewidth='2', label='Admitted')
plt.plot(notAdmitted[0], notAdmitted[1], color='#FFFF00', marker='o', markersize='5', markeredgecolor='black', alpha=1, lineStyle='None', clip_on=False, label='Not admitted')
plt.plot(x1, x2)
plt.legend(loc='upper right', prop={'size': 8})
plt.show()

print(sigmoid(T[0] + T[1]*((45-np.mean(test1Scores))/np.std(test1Scores)) + T[2]*(85-np.mean(test1Scores))/np.std(test1Scores)))
