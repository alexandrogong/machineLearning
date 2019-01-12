# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    # get data
    data = pd.read_csv("./datasets/Bike-Sharing-Dataset/day.csv")
    y_data = data['hum'][:1000]
    x_data = data[['temp', 'atemp']][:1000]
    x0_data = data['temp'][:1000]
    x1_data = data['atemp'][:1000]

    # create model
    model = LinearRegression()
    model.fit(x_data, y_data)

    # log
    print("coefficients:", model.coef_)
    print("intercept:", model.intercept_)

    # plot
    x0_data_new, x1_data_new = np.meshgrid(x0_data, x1_data)
    ax = Axes3D(plt.figure())
    ax.scatter(x0_data, x1_data, y_data, color='r', marker='o')
    z = model.intercept_ + model.coef_[0] * x0_data_new + model.coef_[1] * x1_data_new
    ax.plot_surface(x0_data_new, x1_data_new, z)
    plt.show()