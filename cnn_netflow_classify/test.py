import torch
import torch.nn as nn
import numpy as np
import train


class CNN(nn.Module):
    # cnn结构
    def __init__(self):
        out_channels_1 = 16  # 第一次卷积的卷积核边长
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

    x_train, y_train, x_test, y_test = train.random_divide_dataset(x_data, y_data, k=10)
    train_x = torch.tensor(np.array(x_train))
    train_y = torch.tensor(np.array(y_train))
    test_x = torch.tensor(np.array(x_test))
    test_y = torch.tensor(np.array(y_test))

    cnn = torch.load('cnn_netflow.pkl')

    test_output = cnn(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    test_y_labels = torch.max(test_y, 1)[1].data.numpy()
    print(pred_y[:20], 'prediction number')
    print(test_y_labels[:20], 'real number')

    test_len = test_y.size(0)
    cnt = 0
    for i in range(test_len):
        if pred_y[i] == test_y_labels[i]:
            cnt += 1
    accuracy = cnt / test_len
    print('accuracy', accuracy)
