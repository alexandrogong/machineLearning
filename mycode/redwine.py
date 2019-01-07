# coding utf-8
# @2019 0103
# 机器学习项目
# =============属性==============
# fixed acidity           float64
# volatile acidity        float64
# citric acid             float64
# residual sugar          float64
# chlorides               float64
# free sulfur dioxide     float64
# total sulfur dioxide    float64
# density                 float64
# pH                      float64
# sulphates               float64
# alcohol                 float64
# quality                   int64
# =============属性==============


import sklearn
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split   #  utilities help to choose between model
from sklearn import preprocessing   # utilities for scaling, transforming, and wrangling data
from sklearn.ensemble import RandomForestRegressor  # random forest
from sklearn.pipeline import make_pipeline  # cross validation pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score  # evaluate performance
from sklearn.externals import joblib   #  persist our model for future use


if __name__ == '__main__':
    '''导入数据'''
    filename = "./data/winequality-red.csv"
    dataset = pd.read_csv(filename, ";")

    '''check data format'''
    print(dataset.shape)  # 506条数据 14个特征
    # print(dataset.dtypes)  # 数据类型
    pd.set_option('display.width', 1000)
    # print(dataset.head(3))  # 查看前30行的数据
    # 描述性统计信息
    # pd.set_option('precision', 1)
    # print(dataset.describe())

    # 1. Split data into training and test sets
    y = dataset.quality
    '''separate target from features'''
    X = dataset.drop('quality', axis=1)
    '''split train and test set'''
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123,
                                                        stratify=y)
    print(X_train.shape)

    # 2. Declare data preprocessing steps






