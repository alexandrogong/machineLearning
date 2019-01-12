import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # load data
    data = pd.read_csv("./datasets/Bike-Sharing-Dataset/day.csv")
    x_data = data['temp'][:500]
    y_data = data['atemp'][:500]
    plt.plot(x_data, y_data, 'b.')

    x_data_new = x_data[:, np.newaxis]
    y_data_new = y_data[:, np.newaxis]

    # create modle
    model = LinearRegression()
    model.fit(x_data_new, y_data_new)

    # coefficients
    print("coefficients:", model.coef_)

    # intercept
    print("intercept:", model.intercept_)

    plt.plot(x_data_new, model.predict(x_data_new), 'r')
    plt.show()
