"""
mltools.py

此模块提供了一系列机器学习相关的工具函数和类，
涵盖数据处理、模型训练辅助、数据保存与加载等功能，
旨在简化机器学习项目的开发流程。
"""

import torch
from torch import nn
from torch.utils import data
import re
import json
import time
import httpx
import logging
from tqdm import tqdm
from pathlib import Path
from IPython import display
from datetime import datetime
from collections import Counter
from matplotlib import pyplot as plt


class DataSaveToJson:
    """
    json数据保存器，提供将数据保存到 JSON 文件和从 JSON 文件加载数据的功能。
    """

    @staticmethod
    def save_data(path, label, datas):
        """
        保存数据到指定路径的 JSON 文件中。

        Args:
            path (str): JSON 文件的保存路径。
            label (str): 数据在 JSON 文件中的键名。
            datas: 要保存的数据。
        """
        try:
            with open(path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}
        with open(path, "w") as f:
            data[label] = datas
            json.dump(data, f, indent=4)

    @staticmethod
    def load_data(path, label):
        """
        从指定路径的 JSON 文件中加载数据。

        Args:
            path (str): JSON 文件的路径。
            label (str): 数据在 JSON 文件中的键名。

        Returns:
            从 JSON 文件中加载的数据。
        """
        with open(path, "r") as file:
            return json.load(file)[label]


class Tokenizer:
    """
    分词器，将文本数据转换为词元索引，支持词元与索引之间的相互转换，
    并提供保存和加载词表的功能。
    """

    def __init__(self, datas, min_freq=0):
        """
        初始化分词器。

        Args:
            datas (list[str]): 数据集，包含文本数据。
            min_freq (int, optional): 最小词频，低于该频率的词元将被过滤。默认值为 0。
        """
        tokens = Counter()  # 将文本拆分为词元并统计频率
        for item in datas:
            tokens.update(str(item))
        self.unk = 0  # 未知词元索引为0
        self.cls = 1  # 分类词元索引为1
        self.sep = 2  # 分隔词元索引为2
        self.pad = 3  # 填充词元索引为3
        tokens = [item[0] for item in tokens.items() if item[1] > min_freq]  # 删除低频词元
        self.idx_to_token = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"] + tokens  # 建立词元列表
        # 建立词元字典
        tokens_dict = {value: index + 4 for index, value in enumerate(tokens)}
        self.token_to_idx = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}
        self.token_to_idx.update(tokens_dict)

    def __call__(self, tokens, max_length=None):
        """
        调用分词器，将词元转换为索引。

        Args:
            tokens: 输入的词元，可以是字符串、列表或元组。
            max_length (int, optional): 最大长度，用于填充或截断。默认值为 None。

        Returns:
            torch.Tensor: 转换后的词元索引。
        """
        return self.encode(tokens, max_length)

    def __len__(self):
        """
        返回词表大小。

        Returns:
            int: 词表的长度。
        """
        return len(self.idx_to_token)

    def decode(self, indices):
        """
        根据索引返回词元。

        Args:
            indices (torch.Tensor): 输入的词元索引。

        Returns:
            str 或 list[str]: 解码后的词元。

        Raises:
            TypeError: 如果输入的 indices 不是 torch.Tensor 类型。
        """
        if isinstance(indices, torch.Tensor):
            if indices.dim() == 0:
                return []
            elif indices.dim() == 1:
                return "".join([self.idx_to_token[index] for index in indices.tolist()])
            elif indices.dim() == 2:
                return ["".join([self.idx_to_token[item] for item in index]) for index in indices.tolist()]
        else:
            raise TypeError("indices must be torch.Tensor")

    def encode(self, texts, max_length=None):
        """
        根据词元返回索引。

        Args:
            texts (str 或 list[str] 或 tuple[str]): 输入的词元。
            max_length (int, optional): 最大长度，用于填充或截断。默认值为 None。

        Returns:
            torch.Tensor: 转换后的词元索引。

        Raises:
            TypeError: 如果输入的 texts 不是 str、list[str] 或 tuple[str] 类型。
        """
        if isinstance(texts, str):
            if max_length:
                texts = list(texts)[:max_length] if len(texts) > max_length else list(texts) + ["[PAD]"] * (max_length - len(texts))
            return torch.tensor([self.token_to_idx.get(token, self.unk) for token in texts])
        elif isinstance(texts, (list, tuple)):
            if not max_length:
                max_length = max([len(text) for text in texts])
            return torch.stack([self.encode(text, max_length) for text in texts])
        else:
            raise TypeError(f"texts: {texts}\nThe type of texts is {type(texts)}, while texts must be of type str, tuple[str] or list[str]")

    def save(self, path, label="tokenizer"):
        """
        保存分词器的词表到 JSON 文件。

        Args:
            path (str): JSON 文件的保存路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'tokenizer'。
        """
        DataSaveToJson.save_data(path, label, [self.idx_to_token, self.token_to_idx])

    def load(self, path, label="tokenizer"):
        """
        从 JSON 文件中加载分词器的词表。

        Args:
            path (str): JSON 文件的路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'tokenizer'。
        """
        self.idx_to_token, self.token_to_idx = DataSaveToJson.load_data(path, label)


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
    return (data.DataLoader(_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last) for _data in datas)


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
                with open(f"{save_path}/{file_name}", "wb") as f, tqdm(desc=file_name, total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                    for chuck in response.iter_bytes():
                        f.write(chuck)
                        pbar.update(len(chuck))
    return file_name


class Animator:
    """
    在动画中绘制数据，用于动态展示训练过程中的指标变化。
    """

    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, fmts=None):
        """
        初始化动画器。

        Args:
            xlabel (str, optional): x 轴标签。默认值为 None。
            ylabel (str, optional): y 轴标签。默认值为 None。
            xlim (tuple, optional): x 轴范围。默认值为 None。
            ylim (tuple, optional): y 轴范围。默认值为 None。
            legend (list, optional): 图例。默认值为 None。
            fmts (list, optional): 线条格式。默认值为 None。
        """
        self.fig, self.axes = plt.subplots()  # 生成画布
        self.set_axes = lambda: self.axes.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)  # 初始化设置axes函数
        self.legend = legend  # 图例
        self.fmts = fmts if fmts else ("-", "m--", "g-.", "r:")  # 格式
        plt.close()

    def show(self, Y):
        """
        展示动画。

        Args:
            Y (list): y 轴数据列表。
        """
        X = [list(range(1, len(sublist) + 1)) for sublist in Y]
        self.axes.cla()  # 清除画布
        for x, y, fmt in zip(X, Y, self.fmts):
            self.axes.plot(x, y, fmt)
        self.set_axes()  # 设置axes
        if self.legend:
            self.axes.legend(self.legend)  # 设置图例
        self.axes.grid()  # 设置网格线
        display.display(self.fig)  # 画图
        display.clear_output(wait=True)  # 清除输出

    def save(self, path):
        """
        保存动画为图片文件。

        Args:
            path (str): 图片文件的保存路径。
        """
        self.fig.savefig(path)


def images(images, labels, shape):
    """
    展示图片。

    Args:
        images (torch.Tensor): 图片数据。
        labels (list): 图片标签。
        shape (tuple): 子图布局形状。
    """
    images = images.to(device="cpu")
    fig, axes = plt.subplots(*shape)
    axes = [element for sublist in axes for element in sublist]
    for ax, img, label in zip(axes, images, labels):
        ax.set_title(label)
        ax.set_axis_off()
        ax.imshow(img, cmap="gray")


class Accumulator:
    """
    在 n 个变量上累加，用于统计训练过程中的指标。
    """

    def __init__(self, n):
        """
        初始化累加器。

        Args:
            n (int): 变量个数。
        """
        self.data = [0.0] * n

    def add(self, *args):
        """
        添加数据到累加器。

        Args:
            *args: 要添加的数据。
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """
        重置累加器的数据。
        """
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """
        返回第 n 个累加值。

        Args:
            idx (int): 索引。

        Returns:
            float: 第 idx 个累加值。
        """
        return self.data[idx]


class Recorder:
    """
    n 个记录器，用于记录训练过程中的多个变量的值，支持保存和加载。
    """

    def __init__(self, n):
        """
        初始化记录器。

        Args:
            n (int): 记录器的数量。
        """
        self.data = [[] for _ in range(n)]

    def get_latest_record(self):
        """
        返回最新记录。

        Returns:
            generator: 最新记录的生成器。
        """
        return (item[-1] for item in self.data)

    def max_record_size(self):
        """
        返回最长记录长度。

        Returns:
            int: 最长记录的长度。
        """
        return max((len(item) for item in self.data))

    def reset(self):
        """
        重置记录器的数据。
        """
        self.data = [[] for _ in range(len(self.data))]

    def __getitem__(self, idx):
        """
        返回第 n 个记录器的数据。

        Args:
            idx (int): 索引。

        Returns:
            list: 第 idx 个记录器的数据列表。
        """
        return self.data[idx]

    def save(self, path, label="recorder"):
        """
        保存记录器的数据到 JSON 文件。

        Args:
            path (str): JSON 文件的保存路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'recorder'。
        """
        DataSaveToJson.save_data(path, label, self.data)

    def load(self, path, label="recorder"):
        """
        从 JSON 文件中加载记录器的数据。

        Args:
            path (str): JSON 文件的路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'recorder'。
        """
        self.data = DataSaveToJson.load_data(path, label)


class Timer:
    """
    记录多次运行时间，支持保存和加载记录的时间数据。
    """

    def __init__(self):
        """
        初始化计时器。
        """
        self.times = []

    def start(self):
        """
        启动计时器。
        """
        self.tik = time.time()

    def stop(self):
        """
        停止计时器并将时间记录在列表中。

        Returns:
            float: 本次记录的时间。
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """
        返回平均时间。

        Returns:
            float: 平均时间，单位为秒。如果没有记录时间，则返回 0。
        """
        if self.times:
            return sum(self.times) / len(self.times)
        else:
            return 0

    def sum(self):
        """
        计算记录的所有时间的总和。

        Returns:
            float: 记录的所有时间的总和，单位为秒。如果没有记录时间，则返回 0。
        """
        return sum(self.times)

    @staticmethod
    def str(times):
        """
        将时间转换为格式化的字符串。

        Args:
            times (float): 时间，单位为秒。

        Returns:
            str: 格式化后的时间字符串，格式为 "HH:MM:SS"。
        """
        return time.strftime("%H:%M:%S", time.gmtime(times))

    def save(self, path, label="timer"):
        """
        保存计时器的时间数据到 JSON 文件。

        Args:
            path (str): JSON 文件的保存路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'timer'。
        """
        DataSaveToJson.save_data(path, label, self.times)

    def load(self, path, label="timer"):
        """
        从 JSON 文件中加载计时器的时间数据。

        Args:
            path (str): JSON 文件的路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'timer'。
        """
        self.times = DataSaveToJson.load_data(path, label)


class AutoSaveLoader:
    """
    自动保存加载器，将多个数据的保存和加载功能整合在一起，
    支持添加自定义的保存和加载函数。
    """

    def __init__(self):
        """
        初始化函数，创建保存和加载函数列表。
        """
        self.save_func = []  # 保存函数
        self.load_func = []  # 加载函数

    def add_save_func(self, func):
        """
        添加保存函数。

        Args:
            func (callable): 保存函数。
        """
        self.save_func.append(func)

    def save(self, dir_path):
        """
        保存数据。

        Args:
            dir_path (str): 数据保存的目录路径。
        """
        for func in self.save_func:
            func(dir_path)

    def add_load_func(self, func):
        """
        添加加载函数。

        Args:
            func (callable): 加载函数。
        """
        self.load_func.append(func)

    def load(self, dir_path):
        """
        加载数据。

        Args:
            dir_path (str): 数据加载的目录路径。
        """
        for func in self.load_func:
            func(dir_path)


class Epoch:
    """
    机器学习 Epoch，用于管理训练轮数，支持保存和加载总训练轮数。
    """

    def __init__(self, parent):
        """
        初始化。

        Args:
            parent: 父对象，用于访问日志记录器。
        """
        self._totol_epoch = 0
        self.parent = parent

    def __call__(self, num_epochs):
        """
        返回迭代轮数。

        Args:
            num_epochs (int): 期望的训练轮数。

        Returns:
            int: 本次需要训练的轮数。
        """
        num_epoch = num_epochs - self.totol_epoch if num_epochs > self.totol_epoch else 0  # 计算迭代次数
        self._totol_epoch = max(self.totol_epoch, num_epochs)  # 计算总迭代次数
        # 根据迭代次数产生日志
        self.parent.logger.debug(f"total training epochs {self.totol_epoch}")
        if num_epoch:
            self.parent.logger.debug(f"trained {num_epoch} epochs")
        else:
            self.parent.logger.warning(f"num_epochs is {num_epochs}, less than totol training epoch {self.totol_epoch}, the model won't be trained.")
        return num_epoch

    @property
    def totol_epoch(self):
        """
        返回总迭代次数。

        Returns:
            int: 总训练轮数。
        """
        return self._totol_epoch

    def save(self, path, label="epoch"):
        """
        保存总训练轮数到 JSON 文件。

        Args:
            path (str): JSON 文件的保存路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'epoch'。
        """
        DataSaveToJson.save_data(path, label, self.totol_epoch)

    def load(self, path, label="epoch"):
        """
        从 JSON 文件中加载总训练轮数。

        Args:
            path (str): JSON 文件的路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'epoch'。
        """
        self._totol_epoch = DataSaveToJson.load_data(path, label)


class MachineLearning:
    """
    机器学习工具类，提供批量创建训练辅助对象、管理模型和数据的保存与加载等功能。
    """

    def __init__(self, file_name):
        """
        初始化函数。

        Args:
            file_name (str): 文件名。
        """
        # 定义时间字符串和文件名
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.dir_path = f"../results/{time_str}-{file_name}"
        self.file_name = file_name

        # 创建目录
        Path(self.dir_path).mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger("mylog")
        self.logger.setLevel(logging.DEBUG)
        # 定义日志格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        # 创建文件处理器
        file_handler = logging.FileHandler(f"{self.dir_path}/{self.file_name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 创建自动保存加载器
        self.data_manager = AutoSaveLoader()

    def batch_create(self, create_epoch=True, create_timer=True, create_recorder=True):
        """
        批量创建 Epoch、Timer 和 Recorder 对象。

        Args:
            create_epoch (bool, optional): 是否创建 Epoch 对象。默认值为 True。
            create_timer (bool, optional): 是否创建计时器对象。默认值为 True。
            create_recorder (bool, optional): 是否创建记录器对象。默认值为 True。

        Returns:
            tuple: 包含创建的 Epoch、Timer 和 Recorder 对象的元组，不包含 None 值。
        """
        epoch = self.create_epoch() if create_epoch else None
        timer = self.create_timer() if create_timer else None
        recorder = self.create_recorder(3) if create_recorder else None
        return (item for item in (epoch, timer, recorder) if item is not None)

    def save(self, dir_name=None):
        """
        保存数据。

        Args:
            dir_name (str, optional): 数据保存的目录名。默认值为 None。
        """
        dir_path = f"../results/{dir_name}" if dir_name else self.dir_path
        self.data_manager.save(dir_path)

    def load(self, dir_name=None):
        """
        加载数据。

        Args:
            dir_name (str, optional): 数据加载的目录名。默认值为 None。
        """
        dir_path = f"../results/{dir_name}" if dir_name else self.dir_path
        self.data_manager.load(dir_path)

    def create_epoch(self, label="num_epochs"):
        """
        创建 Epoch 参数。

        Args:
            label (str, optional): Epoch 的标签，建议和被赋值变量名相同。默认值为 'num_epochs'。

        Returns:
            Epoch: 创建的 Epoch 对象。
        """
        epoch = Epoch(self)

        def save(dir_path):
            epoch.save(f"{dir_path}/{self.file_name}.json", label)
            self.logger.debug(f"save Epoch({label}) to {dir_path}/{self.file_name}.json")

        self.data_manager.add_save_func(save)

        def load(dir_path):
            epoch.load(f"{dir_path}/{self.file_name}.json", label)
            self.logger.debug(f"load Epoch({label}) from {dir_path}/{self.file_name}.json")

        self.data_manager.add_load_func(load)

        self.logger.debug(f"create Epoch({label})")
        return epoch

    def create_timer(self, label="timer"):
        """
        创建计时器。

        Args:
            label (str, optional): 计时器的标签，建议和被赋值变量名相同。默认值为 'timer'。

        Returns:
            Timer: 创建的计时器对象。
        """
        timer = Timer()

        def save(dir_path):
            timer.save(f"{dir_path}/{self.file_name}.json", label)
            self.logger.debug(f"save Timer({label}) to {dir_path}/{self.file_name}.json")

        self.data_manager.add_save_func(save)

        def load(dir_path):
            timer.load(f"{dir_path}/{self.file_name}.json", label)
            self.logger.debug(f"load Timer({label}) from {dir_path}/{self.file_name}.json")

        self.data_manager.add_load_func(load)

        self.logger.debug(f"create Timer({label})")
        return timer

    def create_recorder(self, recorder_num, label="recorder"):
        """
        创建记录器。

        Args:
            recorder_num (int): 记录器的数量。
            label (str, optional): 记录器的标签，建议和被赋值变量名相同。默认值为 'recorder'。

        Returns:
            Recorder: 创建的记录器对象。
        """
        recorder = Recorder(recorder_num)

        def save(dir_path):
            recorder.save(f"{dir_path}/{self.file_name}.json", label)
            self.logger.debug(f"save Recorder({label}) to {dir_path}/{self.file_name}.json")

        self.data_manager.add_save_func(save)

        def load(dir_path):
            recorder.load(f"{dir_path}/{self.file_name}.json", label)
            self.logger.debug(f"load Recorder({label}) from {dir_path}/{self.file_name}.json")

        self.data_manager.add_load_func(load)

        self.logger.debug(f"create Recorder({label})")
        return recorder

    def create_animator(self, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, fmts=None, label="animator"):
        """
        创建动画器。

        Args:
            xlabel (str, optional): x 轴标签。默认值为 None。
            ylabel (str, optional): y 轴标签。默认值为 None。
            xlim (tuple, optional): x 轴范围。默认值为 None。
            ylim (tuple, optional): y 轴范围。默认值为 None。
            legend (list, optional): 图例。默认值为 None。
            fmts (list, optional): 格式。默认值为 None。
            label (str, optional): 动画器的标签，建议和被赋值变量名相同。默认值为 'animator'。

        Returns:
            Animator: 创建的动画器对象。
        """
        animator = Animator(xlabel, ylabel, xlim, ylim, legend, fmts)

        def save(dir_path):
            animator.save(f"{dir_path}/{self.file_name}.png")
            self.logger.debug(f"save Animator({label}) to {dir_path}/{self.file_name}.png")

        self.data_manager.add_save_func(save)

        self.logger.debug(f"create Animator({label})")
        return animator

    def add_model(self, model, label="model"):
        """
        添加模型。

        Args:
            model: 模型。
            label (str, optional): 模型的标签，建议和模型变量名相同。默认值为 'model'。
        """
        if not isinstance(model, nn.Module):
            raise RuntimeError(f"model({label}) must be a nn.Module")

        def save(dir_path):
            torch.save(model.state_dict(), f"{dir_path}/{self.file_name}.pth")
            self.logger.debug(f"save model({label}) to {dir_path}/{self.file_name}.pth")

        self.data_manager.add_save_func(save)

        def load(dir_path):
            model.load_state_dict(torch.load(f"{dir_path}/{self.file_name}.pth"))
            self.logger.debug(f"load model({label}) from {dir_path}/{self.file_name}.pth")

        self.data_manager.add_load_func(load)

        self.logger.debug(f"add model({label})")
        self.logger.debug(f"model({label}) is {model}")

    def print_training_time_massage(self, timer, num_epochs, current_epoch):
        """
        打印模型训练时间相关信息，包括已训练时长、平均训练时长和预估剩余训练时长。

        Args:
            timer (Timer): 计时器对象，用于获取训练时间数据。
            num_epochs (int): 总训练轮数。
            current_epoch (int): 当前训练到的轮数。

        Returns:
            None: 此函数仅打印信息，不返回任何值。
        """
        # 计算已训练的总时长，并转换为 HH:MM:SS 格式
        trained_duration = Timer.str(timer.sum())
        # 计算每轮的平均训练时长，并转换为 HH:MM:SS 格式
        average_duration = Timer.str(timer.avg())
        # 计算预估的剩余训练时长，并转换为 HH:MM:SS 格式
        estimated_duration = Timer.str((num_epochs - current_epoch) * timer.avg())
        # 打印训练时间相关信息
        self.logger.info(f"Trained duration: {trained_duration}, Average training duration: {average_duration}, Estimated training duration:{estimated_duration}")

    def model_params(self, model, label="model"):
        """
        打印模型参数数量。

        Args:
            model: 模型对象。

        Returns:
            None: 此函数仅打印信息，不返回任何值。
        """
        if not isinstance(model, nn.Module):
            raise RuntimeError(f"model({label}) must be a nn.Module")
        # 统计模型参数数量
        num_params = sum([param.numel() for param in model.parameters()])
        # 打印模型参数数量
        self.logger.info(f"Number of model({label}) parameters: {num_params / (1000 * 1000):.2f}M")
