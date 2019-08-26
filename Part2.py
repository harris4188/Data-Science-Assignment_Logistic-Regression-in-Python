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
    L = 1
    m = len(y)
    return (-1/m)*(np.transpose(y)*np.log(sigmoid(X*T)) + np.transpose(1-y)*np.log(1 - sigmoid(X*T))) + (L/(2*m))*np.sum(np.power(T, 2))


def gradient_descent(X, T, y):
    alpha = 9
    print(cost(X, T, y))
    m = len(y)
    L = 1
    for x in range(10):
        T[0] = T[0] - (alpha/m)*(np.transpose(X[:, 0])*(sigmoid(X*T) - y))
        T[1:, 0] = T[1:, 0] - (alpha/m)*((np.transpose(X[:, 1:])*(sigmoid(X*T) - y)) + L*T[1:, 0])
        # print(T)
        print(cost(X, T, y))
    return T


def map_features(x1, x2):
    x0 = np.ones(len(y))
    X = np.vstack((x0, x1, x2))
    degree = 6
    for i in range(2, degree + 1):
        for j in range(i+1):
            newFeature = (x1 ** (i-j)) * (x2**j)
            X = np.vstack((X, newFeature))
    X = np.transpose(np.matrix(X))
    return X

data = open('ex2data2.txt').read().splitlines()

test1Results = []
test2Results = []
y = []

for x in range(len(data)):
    data[x] = data[x].split(',')
    test1Results.append(float(data[x][0]))
    test2Results.append(float(data[x][1]))
    y.append(float(data[x][2]))

accepted = [[], []]
rejected = [[], []]

for x in range(len(y)):
    if y[x] == 0:
        rejected[0].append(test1Results[x])
        rejected[1].append(test2Results[x])
    else:
        accepted[0].append(test1Results[x])
        accepted[1].append(test2Results[x])

x1 = np.asarray(test1Results)
x2 = np.asarray(test2Results)

X = map_features(x1, x2)

print(X.shape)

T = np.zeros(28)
print(T)

T = np.transpose(np.matrix(T))

y = np.transpose(np.matrix(y))

#print(cost(X, T, y))

T = gradient_descent(X, T, y)
print(X.shape)
print(T.shape)
print(y.shape)

'''
#Plot Boundary
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i][j] = map_features(np.array(u[i]), np.array(v[j])).dot(np.array(T))

z = z.T
plt.contour(u, v, z, 1)
plt.title('lambda = %f' % 1)
'''
plt.axis([-1, 1.5, -0.8, 1.2])
plt.xlabel('Microchip test 1')
plt.ylabel('Microchip test 2')
plt.plot(accepted[0], accepted[1], 'k+', markeredgewidth='2', label='y = 1')
plt.plot(rejected[0], rejected[1], color='#FFFF00', marker='o', markersize='5', markeredgecolor='black', alpha=1, lineStyle='None', clip_on=False, label='y = 0')
plt.legend(loc='upper right', prop={'size': 8})
plt.show()
