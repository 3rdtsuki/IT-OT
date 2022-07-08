基于CNN的异常流量检测

- DataPrepare.py：预处理pcap流量文件，生成4x4的灰度图，保存为npz文件
- train.py：pytorch实现cnn及训练，保存模型为pkl文件
- test.py：使用训练好的模型测试