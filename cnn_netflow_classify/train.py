import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt


# n个样本中随机选取1/k作为测试集
def random_divide_dataset(x_data, y_data, k):
    n = len(x_data)
    assert n == len(y_data)
    test_indexes = np.random.choice(np.arange(n), size=n // k, replace=False)
    train_indexes = np.delete(np.arange(n), test_indexes)
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in train_indexes:
        x_train.append(x_data[i])
        y_train.append(y_data[i])
    for j in test_indexes:
        x_test.append(x_data[j])
        y_test.append(y_data[j])
    return x_train, y_train, x_test, y_test


class CNN(nn.Module):
    # cnn结构
    def __init__(self):
        out_channels_1 = 16  # 第一次卷积的卷积核数量
        conv1_outsize = 2  # 第一次池化后的图的边长
        out_channels_2 = 32
        conv2_outsize = 1
        super(CNN, self).__init__()
        # 卷积层1的输入图片通道数、卷积核数、卷积核大小3x3、步长、padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels_1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 卷积层2的输入图片通道数、卷积核数、卷积核大小、步长、padding
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_1,
                out_channels=out_channels_2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层，需要一个[in,out]型矩阵相乘，使得输出为10维向量
        self.output = nn.Linear(in_features=out_channels_1 * conv1_outsize * conv1_outsize, out_features=6)
        # self.output = nn.Linear(in_features=out_channels_2 * conv2_outsize * conv2_outsize, out_features=6)

    def forward(self, x):
        # 首先给x增加一个维度，即每个元素i变为[i]
        x = x[:, :, :, np.newaxis]
        # 维度是batch-height-width-channel.
        # 然而，pytorch预计张量维度将以不同的顺序出现：batch-channel-height-width.
        # 使用permute改变维度顺序
        x = x.permute(0, 3, 1, 2)
        x = x.float()  # x必须是float类型，否则报错RuntimeError: expected scalar type Long but found Float
        out = self.conv1(x)
        # out = self.conv2(out)
        out = out.reshape(out.size(0), -1)  # flatten展平
        out = self.output(out)
        return out


if __name__ == "__main__":
    dataset = np.load('dataset.npz')
    x_data = dataset['x']
    y_data = dataset['y']

    # 4:1划分训练集测试集
    x_train, y_train, x_test, y_test = random_divide_dataset(x_data, y_data, k=10)
    train_x = torch.tensor(np.array(x_train))
    train_y = torch.tensor(np.array(y_train))
    test_x = torch.tensor(np.array(x_test))
    test_y = torch.tensor(np.array(y_test))

    EPOCH = 5
    BATCH_SIZE = 50
    LR = 0.001

    train_data = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    cnn = CNN()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, )
    loss_func = nn.CrossEntropyLoss()

    # 一个epoch是一次完整的训练
    for epoch in range(EPOCH):
        # 每个batch
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            b_y = b_y.float()
            # print(output.size(), b_y.size())  # torch.Size([50, 6]) torch.Size([50, 6])
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 如果循环50次，则进行一次测试
            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()  # 获得6维向量的最大维度作为标签
                test_y_labels = torch.max(test_y, 1)[1].data.numpy()
                test_len = test_y.size(0)
                cnt = 0
                for i in range(test_len):
                    if pred_y[i] == test_y_labels[i]:
                        cnt += 1
                accuracy = cnt / test_len
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.6f' % accuracy)

    # 保存模型
    torch.save(cnn, 'cnn_netflow.pkl')
    print('finish training')
