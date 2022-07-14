import numpy as np
import pandas as pd


# 数据预处理
def data_preprocess():
    csv_path = r'..\cnn流量分类\dataset\多分类\malicious_traffic-5.csv'
    df = pd.read_csv(filepath_or_buffer=csv_path, index_col='num')
    # 共17个特征，对每一个特征采用min-max标准化
    for f in range(0, 16):
        feature_name = df.columns[f]
        df[feature_name] = pd.to_numeric(df[feature_name], errors='raise')  # 转为数值型
        n = len(df[feature_name])

        minn = min(df[feature_name])
        maxn = max(df[feature_name])
        delta = maxn - minn
        # min-max标准化
        for i in range(1, n + 1):
            df.loc[i, feature_name] = (df.loc[i, feature_name] - minn) / delta
    output_path = 'output.csv'
    df.to_csv(output_path)


# 为每行流量生成灰度图
def gen_dataset():
    csv_path = 'output.csv'
    df = pd.read_csv(csv_path, index_col='num')
    print(len(df))
    x = []
    y = df['label'].to_numpy()
    for i, line in df.iterrows():
        image = line.to_numpy()[:16].reshape(1,16)  # 选前16个特征
        x.append(image)
    x = np.array(x)
    np.savez('lstm_dataset.npz', x=x, y=y)  # 将灰度图矩阵和标签打包成dataset.npz文件


if __name__ == "__main__":
    data_preprocess()
    gen_dataset()
