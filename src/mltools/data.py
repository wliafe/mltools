from torch.utils import data
import re
import httpx
from tqdm import tqdm
from pathlib import Path


class MyDataset(data.Dataset):
    """
    自定义数据集类，继承自 torch.utils.data.Dataset。
    """

    def __init__(self, datas):
        """
        初始化数据集。

        Args:
            datas: 数据集数据。
        """
        data.Dataset.__init__(self)
        self.data = datas

    def __len__(self):
        """
        返回数据集的长度。

        Returns:
            int: 数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的元素。

        Args:
            idx (int): 数据索引。

        Returns:
            数据集中对应索引的元素。
        """
        return self.data[idx]


def split_data(datas, ratio):
    """
    将数据按比例随机分割。

    Args:
        datas: 待分割的数据。
        ratio (list[float]): 分割比例。

    Returns:
        list: 分割后的数据列表。
    """
    ratio = [r / sum(ratio) for r in ratio]
    nums = [int(len(datas) * r) for r in ratio]
    nums[-1] = len(datas) - sum(nums[:-1])
    return data.random_split(datas, nums)


def iter_data(datas, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False):
    """
    将批量数据转换为迭代器。

    Args:
        datas (list): 数据集列表。
        batch_size (int): 批量大小。
        shuffle (bool, optional): 是否打乱数据。默认值为 True。

    Returns:
        generator: 数据迭代器生成器。
    """
    return (
        data.DataLoader(
            _data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        for _data in datas
    )


def download_file(url, save_path):
    """
    从指定 URL 下载文件。

    Args:
        url (str): 文件的下载 URL。
        save_path (str): 文件的保存路径。

    Returns:
        str: 下载文件的文件名。
    """
    file_name = re.search(r"(?<=/)[^/]+$", url).group()  # 从url中提取文件名
    if not Path(f"{save_path}/{file_name}").exists():  # 如果文件不存在则下载
        Path(save_path).mkdir(parents=True, exist_ok=True)  # 创建保存路径
        with httpx.Client() as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()  # 检查响应状态码
                total_size = int(response.headers.get("Content-Length", 0))  # 获取文件大小
                with open(f"{save_path}/{file_name}", "wb") as f, tqdm(
                    desc=file_name, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
                ) as pbar:
                    for chuck in response.iter_bytes():
                        f.write(chuck)
                        pbar.update(len(chuck))
    return file_name
