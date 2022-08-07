"""
PyCharm Back Propagation
2022.08.06
by SimonYang
"""

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 设计权重w 和 设定w需要计算梯度
w = torch.tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("Predict (before training)", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward() # 反向计算权重这些导数
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()

    print("progress:", epoch, l.item())
print("predict (after training)", 4, forward(4).item())
