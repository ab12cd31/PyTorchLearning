"""
PyCharm Hello(RNN)
2022.08.20
by SimonYang
"""

import torch
import matplotlib.pyplot as plt

num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]
y_data = [3, 1, 2, 3, 2]
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)


net = Model()
critirion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
epoch_list = []
loss_list = []

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = critirion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    loss_list.append(loss.item())
    epoch_list.append(epoch + 1)
    print('idx', idx)
    print('Pridected:', ''.join([idx2char[x] for x in idx]), end='')
    print(',Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.title('Loss', fontsize=20)
plt.ylabel('loss')
plt.grid(ls='--')
plt.show()