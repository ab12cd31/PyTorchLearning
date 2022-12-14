"""
PyCharm 使用不同优化器计算
2022.08.12
by SimonYang
"""

# # SGD
# import torch
# import matplotlib.pyplot as plt
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# epoch_list = [epoch for epoch in range(100)]
# loss_list = []
#
# for epoch in range(100):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#     loss_list.append(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# x_test = torch.tensor([4.0])
# y_test = model(x_test)
# print("y_pred= ", y_test.data.item())
#
# plt.plot(epoch_list, loss_list)
# plt.title('SGD')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# # Adagrad
# import torch
# import matplotlib.pyplot as plt
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
# epoch_list = [epoch for epoch in range(100)]
# loss_list = []
#
# for epoch in range(100):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#     loss_list.append(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# x_test = torch.tensor([4.0])
# y_test = model(x_test)
# print("y_pred= ", y_test.data)
#
# plt.plot(epoch_list, loss_list)
# plt.title('Adagrad')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# # Adam
# import torch
# import matplotlib.pyplot as plt
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# epoch_list = [epoch for epoch in range(100)]
# loss_list = []
#
# for epoch in range(100):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#     loss_list.append(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# x_test = torch.tensor([4.0])
# y_test = model(x_test)
# print("y_pred= ", y_test.data)
#
# plt.plot(epoch_list, loss_list)
# plt.title('Adam')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# # Adamax
# import torch
# import matplotlib.pyplot as plt
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.1)
# epoch_list = [epoch for epoch in range(100)]
# loss_list = []
#
# for epoch in range(100):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#     loss_list.append(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# x_test = torch.tensor([4.0])
# y_test = model(x_test)
# print("y_pred= ", y_test.data.item())
#
# plt.plot(epoch_list, loss_list)
# plt.title('Adamax')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# # ASGD
# import torch
# import matplotlib.pyplot as plt
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)
# epoch_list = [epoch for epoch in range(100)]
# loss_list = []
#
# for epoch in range(100):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#     loss_list.append(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# x_test = torch.tensor([4.0])
# y_test = model(x_test)
# print("y_pred= ", y_test.data.item())
#
# plt.plot(epoch_list, loss_list)
# plt.title('ASGD')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# # LBFGS
# import torch
# import matplotlib.pyplot as plt
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
# epoch_list = []
# loss_list = []
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
#
#
# def closure():
#     optimizer.zero_grad()
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch,loss.item())
#     loss_list.append(loss.item())
#     epoch_list.append(epoch)
#     loss.backward()
#     return loss
#
#
# for epoch in range(100):
#     optimizer.step(closure)
#
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# x_test = torch.tensor([4.0])
# y_test = model(x_test)
# print("y_pred= ", y_test.data.item())
#
# print(epoch_list)
# plt.plot(epoch_list, loss_list)
# plt.title('LBFGS')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# # RMSprop
# import torch
# import matplotlib.pyplot as plt
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
# epoch_list = [epoch for epoch in range(100)]
# loss_list = []
#
# for epoch in range(100):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#     loss_list.append(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# x_test = torch.tensor([4.0])
# y_test = model(x_test)
# print("y_pred= ", y_test.data.item())
#
# plt.plot(epoch_list, loss_list)
# plt.title('Rmsprop')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# # Rprop
# import torch
# import matplotlib.pyplot as plt
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
# epoch_list = [epoch for epoch in range(100)]
# loss_list = []
#
# for epoch in range(100):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#     loss_list.append(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# x_test = torch.tensor([4.0])
# y_test = model(x_test)
# print("y_pred= ", y_test.data.item())
#
# plt.plot(epoch_list, loss_list)
# plt.title('Rprop')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# CSDN上一种七种算法都展示的图
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#实现定义好数据集的x值和y值
x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])


#构建训练模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()#继承父类
        self.linear = torch.nn.Linear(1,1)#定义一个线性计算单元，继承Module块，也可以自主实现反向传播

    def forward(self,x):
        y_pred = self.linear(x)#可调用对象，进行了相应的forward计算
        return y_pred

model1 = LinearModel()  # 定义的模型结构初始化
model2 = LinearModel()
model3 = LinearModel()
model4 = LinearModel()
model5 = LinearModel()
#model6 = LinearModel()
model7 = LinearModel()
model8 = LinearModel()
models = [model1,model2,model3,model4,model5,model7,model8]
#构建损失函数的计算和优化器
criterion = torch.nn.MSELoss(size_average = False)

op1 = torch.optim.SGD(model1.parameters(),lr = 0.01)#以下分别尝试不同的优化器
op2 = torch.optim.Adagrad(model2.parameters(),lr = 0.01)
op3 = torch.optim.Adam(model3.parameters(),lr = 0.01)
op4 = torch.optim.Adamax(model4.parameters(),lr = 0.01)
op5 = torch.optim.ASGD(model5.parameters(),lr = 0.01)
#op6 = torch.optim.LBFGS(model6.parameters(),lr = 0.01)
op7 = torch.optim.RMSprop(model7.parameters(),lr = 0.01)
op8 = torch.optim.Rprop(model8.parameters(),lr = 0.01)
ops = [op1,op2,op3,op4,op5,op7,op8]

titles = ['SGD','Adagrad','Adam','Adamax','ASGD','RNSprop','Rprop']

fig,ax = plt.subplots(2,4,figsize = (20,10),dpi = 100)
#ax[0][0].imshow()
#训练过程，对每个优化器都进行尝试
index = 0
for item in zip(ops,models):#分别尝试每个优化器
    epochs = []
    costs = []

    model = item[1]
    op = item[0]

    for epoch in range(100):
        epochs.append(epoch)
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        costs.append(loss.item())
        print(epoch, loss)

        op.zero_grad()
        loss.backward()
        op.step()

    # 对训练结果进行输出
    print('w = ', model.linear.weight.item())
    print('b = ', model.linear.bias.item())

    # 进行模型测试
    x_test = torch.Tensor([[4.0]])
    y_test = model(x_test)
    print('y_pred = ', y_test.data)

    # 训练过程可视化
    #plt.subplots(2,1,index)
    a = int(index /4)
    b = int(index % 4)
    ax[a][b].set_ylabel('Cost')
    ax[a][b].set_xlabel('Epoch')
    ax[a][b].set_title(titles[index])
    ax[a][b].plot(epochs, costs)
    index += 1
plt.show()
