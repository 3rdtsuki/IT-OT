# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.utils.data as Data


# LSTM结构
class LSTM(nn.Module):
    """
        Parameters：
        - input_size: feature size，输入特征数
        - hidden_size: number of hidden units，一个隐藏层的神经元数，即h_t的维度，一般是2的n次幂
        - output_size: number of output，输出特征数
        - num_layers: layers of LSTM to stack，隐藏层数，1~2
    """

    def __init__(self, input_size, hidden_size=16, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        _x = _x.float()  # _x is input, size (batch_num, seq_len, input_size)
        x, _ = self.lstm(_x)
        s, b, h = x.shape  # x is output, size (batch_num, seq_len, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


def divide_dataset(x_data, y_data, k):
    n = len(x_data)
    print(n)
    gap = n // k * (k - 1)
    x_train, y_train = x_data[:gap], y_data[:gap]
    x_test, y_test = x_data[gap:], y_data[gap:]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    dataset = np.load('lstm_dataset.npz')
    x_data = dataset['x'][:3000]
    y_data = dataset['y'][:3000]

    x_train, y_train, x_test, y_test = divide_dataset(x_data, y_data, k=10)
    train_x = torch.tensor(np.array(x_train))
    train_y = torch.tensor(np.array(y_train))
    test_x = torch.tensor(np.array(x_test))
    test_y = torch.tensor(np.array(y_test))

    INPUT_FEATURES_NUM = 16
    OUTPUT_FEATURES_NUM = 6
    EPOCH = 100
    BATCH_SIZE = len(train_x)  # 注意：输入lstm的序列是训练集的所有流量，所以batch_size = len(训练集)
    LR = 0.001

    train_data = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

    lstm = LSTM(input_size=INPUT_FEATURES_NUM,
                hidden_size=16,
                output_size=OUTPUT_FEATURES_NUM,
                num_layers=1)
    print('LSTM model:', lstm)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = lstm(b_x).reshape(-1, OUTPUT_FEATURES_NUM)  # output必须从3维转为2维
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = lstm(test_x).reshape(-1, OUTPUT_FEATURES_NUM)  # output必须从3维转为2维
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.6f' % accuracy)

    # torch.save(lstm,'lstm_netflow.pkl')
