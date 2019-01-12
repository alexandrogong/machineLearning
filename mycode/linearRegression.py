# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt


# cost function
def compute_error(b, k, x_data, y_data):
    total_error = 0
    for i in range(len(x_data)):
        total_error += (y_data[i]-(k*x_data[i]+b))**2
    return total_error/float(len(x_data))


# gradient decent function
def gradient_decent_runner(x_data, y_data, b, k, lr, epochs):
    m = float(len(x_data))
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        for j in range(len(x_data)):
            b_grad += (1/m) * ((k*x_data[j]+b)-y_data[j])
            k_grad += (1/m) * x_data[j]*((k*x_data[j]+b)-y_data[j])
        b = b-lr*b_grad
        k = k-lr*k_grad
        # plot every 5 interations
        if i % 100 == 0:
            print("epochs:", i)
            plt.plot(x_data, k * x_data + b, 'r')
    return b, k


if __name__ == "__main__":
    # set learning rate
    lr = 0.1
    # set intercept
    b = 0
    # slope
    k = 0.5
    # max interations
    epochs = 1000

    # load data
    data = pd.read_csv("./datasets/Bike-Sharing-Dataset/day.csv")
    # print(data[:10])
    x_data = data['temp'][:500]
    y_data = data['atemp'][:500]

    plt.plot(x_data, y_data, 'b.')

    error = compute_error(b, k, x_data, y_data)
    print("Starting b={0}, k={1}, error={2}".format(b, k, error))
    print('running...')
    b, k = gradient_decent_runner(x_data, y_data, b, k, lr, epochs)
    error = compute_error(b, k, x_data, y_data)
    print('After {0}, interation b = {1}, k={2}, error={3}'.format(epochs, b, k, error))

    # plot
    plt.show()


