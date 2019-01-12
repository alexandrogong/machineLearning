# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_error(theta0, theta1, theta2, x_data, y_data):
    total_error = 0
    for i in range(len(y_data)):
        total_error += (y_data[i]-(theta0+theta1*x_data.ix[i, 0]+theta2*x_data.ix[i, 1]))**2
    return total_error/len(y_data)


def gradient_decrease(lr, epochs, theta0, theta1, theta2, x_data, y_data):
    m = len(y_data)
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        for j in range(m):
            print(j)
            theta0_grad += (-1 / m) * (y_data[j] - (theta0 + theta1 * x_data.ix[j, 0] + theta2 * x_data.ix[j, 1]))
            theta1_grad += (-1 / m) * (y_data[j] - (theta0 + theta1 * x_data.ix[j, 0] + theta2 * x_data.ix[j, 1]))*x_data.ix[j, 0]
            theta2_grad += (-1 / m) * (y_data[j] - (theta0 + theta1 * x_data.ix[j, 0] + theta2 * x_data.ix[j, 1]))*x_data.ix[j, 1]
        theta0 = theta0 - lr * theta0_grad
        theta1 = theta1 - lr * theta1_grad
        theta2 = theta2 - lr * theta2_grad

        if i % 100 == 0:
            pass

    return theta0, theta1, theta2


if __name__ == '__main__':
    # set paras
    lr = 0.1
    epochs = 100
    theta0 = 1
    theta1 = 1
    theta2 = 1

    # get data
    data = pd.read_csv("./datasets/Bike-Sharing-Dataset/day.csv")
    y_data = data['hum'][:1000]
    x_data = data[['temp', 'atemp']][:1000]
    x0_data = data['temp'][:1000]
    x1_data = data['atemp'][:1000]

    # gradiant decent
    theta0, theta1, theta2 = gradient_decrease(lr, epochs, theta0, theta1, theta2, x_data, y_data)

    # log
    print("theta0:", theta0)
    print("theta1:", theta1)
    print("theta2:", theta2)
    print('cost:', compute_error(theta0, theta1, theta2, x_data, y_data))

    # generate grid
    x0_data_new, x1_data_new = np.meshgrid(x0_data, x1_data)

    # plot
    ax = Axes3D(plt.figure())
    ax.scatter(x0_data, x1_data, y_data, color='r', marker='o')
    z = theta0 + theta1*x0_data_new + theta1*x1_data_new
    ax.plot_surface(x0_data_new, x1_data_new, z)
    plt.show()


