import mltools
import pandas as pd
from pathlib import Path
from torchvision import transforms, datasets


def mnist(path='../data', batch_size=100):
    """
    加载数据集 MNIST。

    Args:
        path (str, optional): 数据集路径。默认值为 '../data'。
        batch_size (int, optional): 批量大小。默认值为 100。

    Returns:
        tuple: 训练集、验证集、测试集的迭代器。
    """
    download = False if Path(f'{path}/MNIST').exists() else True
    trans = transforms.ToTensor()  # 数据集格式转换
    train_data = datasets.MNIST(root=path, train=True, transform=trans, download=download)
    test_data = datasets.MNIST(root=path, train=False, transform=trans, download=download)
    train_data, val_data = mltools.split_data(train_data, [9, 1])  # 训练集和验证集比例9：1
    return mltools.iter_data([train_data, val_data, test_data], batch_size)  # 返回数据迭代器


def chn_senti_corp(path='../data', batch_size=100):
    """
    加载数据集 ChnSentiCorp。

    Args:
        path (str, optional): 数据集路径。默认值为 '../data'。
        batch_size (int, optional): 批量大小。默认值为 100。

    Returns:
        tuple: 训练集、验证集、测试集的迭代器和分词器。
    """
    url = 'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/refs/heads/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv'
    file_name = mltools.download_file(url, path)
    chn_senti_corp = pd.read_csv(f'{path}/{file_name}')  # 读数据集
    chn_senti_corp_data = [(str(item.review), item.label) for item in chn_senti_corp.itertuples()]
    chn_senti_corp_data = mltools.MyDataset(chn_senti_corp_data)  # 生成Dataset
    train_data, val_data, test_data = mltools.split_data(chn_senti_corp_data, [0.7, 0.15, 0.15])  # 划分训练集、验证集、测试集
    train_iter, val_iter, test_iter = mltools.iter_data([train_data, val_data, test_data], batch_size)  # 产生迭代器
    tokenizer = mltools.Tokenizer(chn_senti_corp.iloc[:, 1].values, min_freq=10)  # 建立分词器
    return train_iter, val_iter, test_iter, tokenizer  # 返回迭代器和分词器
