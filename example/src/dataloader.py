import mltools
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
