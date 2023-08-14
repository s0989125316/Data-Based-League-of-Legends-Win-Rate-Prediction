import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
# 创建自定义缩放器类(标准化)
class CustomScaler(BaseEstimator, TransformerMixin):

    # 声明一些基本内容和信息
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        # scaler是Standard Scaler对象
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    # 基于StandardScale的拟合方法

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    # 进行实际缩放的变换方法

    def transform(self, X, y=None, copy=None):
        # 记录列的初始顺序
        init_col_order = X.columns

        # 缩放创建类实例时选择的所有功能
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        # 声明一个包含所有未缩放信息的变量
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]

        # 返回包含所有已缩放要素和所有未缩放要素的数据框
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]



