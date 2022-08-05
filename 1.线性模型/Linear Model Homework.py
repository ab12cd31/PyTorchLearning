"""
PyCharm Linear Model Homework
2022.08.05
by SimonYang
"""

# 加上参数b 变成两个未知数的线性模型,画三维图
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


w_list = np.arange(0.0, 4.1, 0.1)
b_list = np.arange(-2.0, 2.1, 0.1)
[w, b] = np.meshgrid(w_list, b_list)
mse_list = []

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("w", fontsize=10, labelpad=15)
ax.set_ylabel("b", fontsize=10, labelpad=15)
ax.set_zlabel("Loss", fontsize=15, labelpad=15)
ax.text(0.2, 2, 43, "Cost Value", color='black')
ax.plot_surface(w, b, l_sum / 3, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()
