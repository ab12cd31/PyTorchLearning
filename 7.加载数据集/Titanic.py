"""
PyCharm Titanic
2022.08.14
by SimonYang
"""

import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader


class SurviveDataset(Dataset):

    def __init__(self, filepath):
        x = np.loadtxt(filepath, delimiter=',', dtype=np.float32, usecols=(2, 5, 6, 7, 8, 10, 12))
        # 上面只取有效特征，类似人名，票号等唯一特征对训练没用就没取。
        y = np.loadtxt(filepath, delimiter=',', dtype=np.float32, usecols=1)
        # 'delimiter'为分隔符
        y = y[:, np.newaxis]
        # 这里增加一维，不然计算loss的时候维度不同会报错

        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = SurviveDataset('train.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.Linear1 = torch.nn.Linear(7, 5)
        self.Linear2 = torch.nn.Linear(5, 3)
        self.Linear3 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.Linear1(x))
        x = self.sigmoid(self.Linear2(x))
        x = self.sigmoid(self.Linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
