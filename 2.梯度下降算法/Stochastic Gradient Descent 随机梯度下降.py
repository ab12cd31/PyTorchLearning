"""
PyCharm Stochastic Gradient Descent 随机梯度下降.py
2022.08.06
by SimonYang
"""
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def loss(xs, ys):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def grident(xs, ys):
    return 2 * x * (x * w - y)


epoch_list = []
Cost_list = []
print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = grident(x, y)
        w = w - 0.01 * grad
        print("\tgrad:", x, y, grad)
        l = loss(x_data, y_data)
        epoch_list.append(epoch)
        Cost_list.append(l)
        print("progress:", epoch, 'w=', w, "loss=" , l)
    print("Predict (after training)", 4, forward(4))

plt.plot(epoch_list, Cost_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
