import torch
from torch.utils import data
import re
import yaml
import httpx
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
from mltools import utils

"""
数据集处理
"""


class Tokenizer:
    """
    分词器，将文本数据转换为词元索引，支持词元与索引之间的相互转换，并提供保存和加载词表的功能。
    """

    def __init__(self, datas: list[str], min_freq: int = 0):
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

    def __call__(self, tokens: str | list[str] | tuple[str], max_length: int = None) -> torch.Tensor:
        """
        调用分词器，将词元转换为索引。

        Args:
            tokens (str 或 list[str] 或 tuple[str]): 输入的词元。
            max_length (int, optional): 最大长度，用于填充或截断。默认值为 None。

        Returns:
            torch.Tensor: 转换后的词元索引。
        """
        return self.encode(tokens, max_length)

    def __len__(self) -> int:
        """
        返回词表大小。

        Returns:
            int: 词表的长度。
        """
        return len(self.idx_to_token)

    def decode(self, indices: torch.Tensor) -> str | list[str]:
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
            raise TypeError("indices 必须是 torch.Tensor 类型")

    def encode(self, texts: str | list[str] | tuple[str], max_length: int = None) -> torch.Tensor:
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
                texts = (
                    list(texts)[:max_length]
                    if len(texts) > max_length
                    else list(texts) + ["[PAD]"] * (max_length - len(texts))
                )
            return torch.tensor([self.token_to_idx.get(token, self.unk) for token in texts])
        elif isinstance(texts, (list, tuple)):
            if not max_length:
                max_length = max([len(text) for text in texts])
            return torch.stack([self.encode(text, max_length) for text in texts])
        else:
            raise TypeError(
                f"texts: {texts}\nThe type of texts is {type(texts)}, while texts must be of type str, tuple[str] or list[str]"
            )

    def save(self, path: str, label: str = "tokenizer"):
        """
        保存分词器的词表到 JSON 文件。

        Args:
            path (str): JSON 文件的保存路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'tokenizer'。
        """
        utils.DataSaveToJson.save_data(path, label, [self.idx_to_token, self.token_to_idx])

    def load(self, path: str, label: str = "tokenizer"):
        """
        从 JSON 文件中加载分词器的词表。

        Args:
            path (str): JSON 文件的路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'tokenizer'。
        """
        self.idx_to_token, self.token_to_idx = utils.DataSaveToJson.load_data(path, label)


class MyDataset(data.Dataset):
    """
    自定义数据集类，继承自 torch.utils.data.Dataset，用于管理机器学习任务中的数据。
    """

    def __init__(self, datas: list):
        """
        初始化数据集

        Args:
            datas (list): 数据集内容，可以是任何格式的数据
        """
        data.Dataset.__init__(self)
        self.data = datas

    def __len__(self) -> int:
        """
        返回数据集的样本数量

        Returns:
            int: 数据集中的样本总数
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> any:
        """
        根据索引获取单个数据样本

        Args:
            idx (int): 数据索引

        Returns:
            对应索引的数据样本
        """
        return self.data[idx]


def split_data(datas: list, ratio: list) -> list:
    """
    划分数据集

    Args:
        datas (list): 数据集内容，可以是任何格式的数据
        ratio (list): 划分比例，例如 [0.8, 0.2] 表示划分成 80% 训练集和 20% 测试集

    Returns:
        list: 划分后的数据集，每个元素都是一个数据集
    """
    ratio = [r / sum(ratio) for r in ratio]
    nums = [int(len(datas) * r) for r in ratio]
    nums[-1] = len(datas) - sum(nums[:-1])
    return data.random_split(datas, nums)


def iter_data(
    datas: list,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> data.DataLoader:
    """
    迭代数据集

    Args:
        datas (list): 数据集内容，可以是任何格式的数据
        batch_size (int): 每个批次的样本数量
        shuffle (bool, optional): 是否在每个 epoch 开始时打乱数据. 默认值为 True.
        num_workers (int, optional): 用于数据加载的子进程数量. 默认值为 0.
        pin_memory (bool, optional): 是否将数据加载到 CUDA 固定内存中. 默认值为 False.
        drop_last (bool, optional): 是否丢弃最后一个批次, 如果数据集大小不能被批次大小整除. 默认值为 False.

    Returns:
        data.DataLoader: 数据加载器，用于迭代数据集
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


"""
数据集下载
"""


def download_file(url: str, *, save_path: str) -> str:
    """
    下载文件

    Args:
        url (str): 文件的 URL 地址
        save_path (str): 保存文件的路径

    Returns:
        str: 下载的文件名
    """
    file_name = re.search(r"(?<=/)[^/]+$", url).group()  # 从url中提取文件名
    if not Path(f"{save_path}/{file_name}").exists():  # 如果文件不存在则下载
        Path(save_path).mkdir(parents=True, exist_ok=True)  # 创建保存路径
        with httpx.Client() as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()  # 检查响应状态码
                total_size = int(response.headers.get("Content-Length", 0))  # 获取文件大小
                with (
                    open(f"{save_path}/{file_name}", "wb") as f,
                    tqdm(desc=file_name, total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar,
                ):
                    for chuck in response.iter_bytes():
                        f.write(chuck)
                        pbar.update(len(chuck))
    return file_name


"""
目标检测数据处理
"""


class BaseBbox:
    """
    基础边界框类，用于表示物体的边界框。
    """

    def __init__(self, bbox: list, *, bbox_type: str = "xmin_ymin_xmax_ymax"):
        """
        初始化边界框

        Args:
            bbox (list): 边界框参数，格式根据 bbox_type 不同而不同
            bbox_type (str, optional): 边界框格式，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h"、"center_w_h". 默认值为 "xmin_ymin_xmax_ymax".

        Raises:
            ValueError: 如果类别不是整数
            ValueError: 如果 bbox 参数不归一化
            ValueError: 如果 bbox_type 不是 'xmin_ymin_xmax_ymax'、'xmin_ymin_w_h' 或 'center_w_h'
        """
        if isinstance(bbox[0], int):
            self.class_id = bbox[0]
        else:
            raise ValueError("类别必须是整数")
        if all(isinstance(item, float) for item in bbox[1:5]):
            self.bbox = bbox[1:5]
        else:
            raise ValueError("bbox 参数必须归一化")
        if bbox_type in ["xmin_ymin_xmax_ymax", "xmin_ymin_w_h", "center_w_h"]:
            self.bbox_type = bbox_type
        else:
            raise ValueError("bbox_type 必须是 'xmin_ymin_xmax_ymax'、'xmin_ymin_w_h' 或 'center_w_h'")

    def __str__(self) -> str:
        """
        返回边界框的字符串表示

        Returns:
            str: 边界框的字符串表示，格式为 "{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
        """
        return f"{self.class_id} {self.bbox[0]} {self.bbox[1]} {self.bbox[2]} {self.bbox[3]}"

    def __repr__(self) -> str:
        """
        返回边界框的字符串表示

        Returns:
            str: 边界框的字符串表示，格式为 "BaseBbox(class_id={self.class_id}, bbox=[{self.bbox[0]}, {self.bbox[1]}, {self.bbox[2]}, {self.bbox[3]}])"
        """
        return f"BaseBbox(class_id={self.class_id}, bbox=[{self.bbox[0]}, {self.bbox[1]}, {self.bbox[2]}, {self.bbox[3]}], bbox_type={self.bbox_type})"

    def to_list(self) -> list:
        """
        返回边界框的坐标表示

        Returns:
            list: 边界框的坐标表示，格式为 [class_id, x_min, y_min, x_max, y_max]
        """
        return [self.class_id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]]

    def xmin_ymin_xmax_ymax(self):
        """
        返回边界框的坐标表示

        Returns:
            BaseBbox: 格式为 xmin_ymin_xmax_ymax 的边界框的坐标表示。
        """
        if self.bbox_type == "xmin_ymin_xmax_ymax":
            return self
        elif self.bbox_type == "xmin_ymin_w_h":
            return BaseBbox(
                [self.class_id, self.bbox[0], self.bbox[1], self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]],
                bbox_type="xmin_ymin_xmax_ymax",
            )
        elif self.bbox_type == "center_w_h":
            return BaseBbox(
                [
                    self.class_id,
                    self.bbox[0] - self.bbox[2] / 2,
                    self.bbox[1] - self.bbox[3] / 2,
                    self.bbox[0] + self.bbox[2] / 2,
                    self.bbox[1] + self.bbox[3] / 2,
                ],
                bbox_type="xmin_ymin_xmax_ymax",
            )
        else:
            raise ValueError("bbox_type 必须是 'xmin_ymin_xmax_ymax'、'xmin_ymin_w_h' 或 'center_w_h'")

    def xmin_ymin_w_h(self):
        """
        返回边界框的坐标表示

        Returns:
            BaseBbox: 格式为 xmin_ymin_w_h 的边界框的坐标表示。
        """
        if self.bbox_type == "xmin_ymin_w_h":
            return self
        else:
            bbox = self.xmin_ymin_xmax_ymax().to_list()
            return BaseBbox(
                [bbox[0], bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]],
                bbox_type="xmin_ymin_w_h",
            )

    def center_w_h(self):
        """
        返回边界框的坐标表示

        Returns:
            BaseBbox: 格式为 center_w_h 的边界框的坐标表示。
        """
        if self.bbox_type == "center_w_h":
            return self
        else:
            bbox = self.xmin_ymin_xmax_ymax().to_list()
            return BaseBbox(
                [
                    bbox[0],
                    bbox[1] + (bbox[3] - bbox[1]) / 2,
                    bbox[2] + (bbox[4] - bbox[2]) / 2,
                    bbox[3] - bbox[1],
                    bbox[4] - bbox[2],
                ],
                bbox_type="center_w_h",
            )

    def convert(self, bbox_type: str):
        """
        转换边界框格式

        Args:
            bbox_type (str): 目标边界框格式，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"

        Returns:
            BaseBbox: 转换后的边界框实例

        Raises:
            ValueError: 如果 bbox_type 不是 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h" 中的一个
        """
        if bbox_type == "xmin_ymin_xmax_ymax":
            return self.xmin_ymin_xmax_ymax()
        elif bbox_type == "xmin_ymin_w_h":
            return self.xmin_ymin_w_h()
        elif bbox_type == "center_w_h":
            return self.center_w_h()
        else:
            raise ValueError("bbox_type 必须是 'xmin_ymin_xmax_ymax'、'xmin_ymin_w_h' 或 'center_w_h'")

    @staticmethod
    def normalize(bbox: list, *, width: int, height: int) -> list:
        """
        归一化边界框坐标

        Args:
            bbox (list): 边界框参数，格式为 [class_id, x_min, y_min, x_max, y_max]
            width (int): 图片宽度
            height (int): 图片高度

        Returns:
            list: 归一化后的边界框参数，格式为 [class_id, x_min / width, y_min / height, x_max / width, y_max / height]
        """
        return [bbox[0], bbox[1] / width, bbox[2] / height, bbox[3] / width, bbox[4] / height]

    @staticmethod
    def unnormalize(bbox: list, *, width: int, height: int) -> list:
        """
        反归一化边界框坐标

        Args:
            bbox (list): 归一化后的边界框参数，格式为 [class_id, x_min / width, y_min / height, x_max / width, y_max / height]
            width (int): 图片宽度
            height (int): 图片高度

        Returns:
            list: 反归一化后的边界框参数，格式为 [class_id, int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)]
        """
        return [bbox[0], int(bbox[1] * width), int(bbox[2] * height), int(bbox[3] * width), int(bbox[4] * height)]


class Bbox:
    """
    边界框容器类
    """

    def __init__(self, bboxes: list = None, bbox_type: str = "xmin_ymin_xmax_ymax"):
        """
        初始化 Bbox 实例

        Args:
            bboxes (list): 边界框列表，每个元素为 BaseBbox 实例
            bbox_type (str, optional): 边界框类型，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"，默认为 "xmin_ymin_xmax_ymax"

        Raises:
            ValueError: 如果 bboxes 参数不是列表
            ValueError: 如果 bboxes 列表元素不是列表或 BaseBbox 实例
            ValueError: 如果 bboxes 列表元素类型与 bbox_type 参数不匹配
        """
        if bboxes is None:
            bboxes = []
        if not isinstance(bboxes, list):
            raise ValueError("bboxes 参数必须是列表")
        if not all(isinstance(bbox, list | BaseBbox) for bbox in bboxes) and len(bboxes) != 0:
            raise ValueError("bboxes 列表元素必须是列表或 BaseBbox 实例")
        self.bbox_type = bbox_type
        self.bboxes = []
        for bbox in bboxes:
            if isinstance(bbox, BaseBbox):
                if bbox.bbox_type == bbox_type:
                    self.bboxes.append(bbox)
                else:
                    raise ValueError(f"bboxes 列表元素 {bbox} 类型与 bbox_type 参数 {bbox_type} 不匹配")
            else:
                self.bboxes.append(BaseBbox(bbox, bbox_type=bbox_type))

    def __getitem__(self, index: int) -> BaseBbox:
        """
        获取指定索引的边界框

        Args:
            index (int): 边界框索引

        Returns:
            BaseBbox: 指定索引的边界框实例
        """
        return self.bboxes[index]

    def __len__(self) -> int:
        """
        返回边界框列表的长度

        Returns:
            int: 边界框列表的长度
        """
        return len(self.bboxes)

    def __str__(self) -> str:
        """
        返回边界框列表的字符串表示

        Returns:
            str: 边界框列表的字符串表示
        """
        return "\n".join(str(bbox) for bbox in self.bboxes)

    def __repr__(self) -> str:
        """
        返回边界框列表的字符串表示

        Returns:
            str: 边界框列表的字符串表示
        """
        return "Bbox([\n\t" + ",\n\t".join(str(bbox.__repr__()) for bbox in self.bboxes) + ",\n])"

    def to_list(self) -> list:
        """
        返回边界框列表的坐标表示

        Returns:
            list: 边界框列表的坐标表示
        """
        return [bbox.to_list() for bbox in self.bboxes]

    def append(self, bbox: BaseBbox | list):
        """
        在边界框列表末尾添加一个边界框

        Args:
            bbox (BaseBbox | list): 要添加的边界框实例
        """
        if isinstance(bbox, list):
            self.bboxes.append(BaseBbox(bbox, bbox_type=self.bbox_type))
        elif isinstance(bbox, BaseBbox):
            if bbox.bbox_type == self.bbox_type:
                self.bboxes.append(bbox)
            else:
                raise ValueError(f"bbox 参数 {bbox} 类型与 bbox_type 参数 {self.bbox_type} 不匹配")
        else:
            raise ValueError(f"bbox 参数 {bbox} 类型必须为 list 或 BaseBbox")

    def xmin_ymin_xmax_ymax(self):
        """
        返回边界框的坐标表示

        Returns:
            Bbox: 格式为 xmin_ymin_xmax_ymax 边界框列表的坐标表示。
        """
        if self.bbox_type == "xmin_ymin_xmax_ymax":
            return self
        else:
            return Bbox([bbox.xmin_ymin_xmax_ymax() for bbox in self.bboxes], bbox_type="xmin_ymin_xmax_ymax")

    def xmin_ymin_w_h(self):
        """
        返回边界框的坐标表示

        Returns:
            Bbox: 格式为 xmin_ymin_w_h 边界框列表的坐标表示。
        """
        if self.bbox_type == "xmin_ymin_w_h":
            return self
        else:
            return Bbox([bbox.xmin_ymin_w_h() for bbox in self.bboxes], bbox_type="xmin_ymin_w_h")

    def center_w_h(self):
        """
        返回边界框的坐标表示

        Returns:
            Bbox: 格式为 center_w_h 边界框列表的坐标表示。
        """
        if self.bbox_type == "center_w_h":
            return self
        else:
            return Bbox([bbox.center_w_h() for bbox in self.bboxes], bbox_type="center_w_h")

    def convert(self, bbox_type: str):
        """
        转换边界框格式

        Args:
            bbox_type (str): 目标边界框格式，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"

        Returns:
            BaseBbox: 转换后的边界框实例
        """
        return Bbox([bbox.convert(bbox_type) for bbox in self.bboxes], bbox_type=bbox_type)

    @staticmethod
    def normalize(bboxes: list, *, width: int, height: int) -> list:
        """
        归一化边界框坐标

        Args:
            bboxes (list): 边界框列表，每个元素为 BaseBbox 实例
            width (int): 图片宽度
            height (int): 图片高度

        Returns:
            list: 归一化后的边界框列表，每个元素为 BaseBbox 实例
        """
        return [BaseBbox.normalize(bbox, width=width, height=height) for bbox in bboxes]

    @staticmethod
    def unnormalize(bboxes: list, *, width: int, height: int) -> list:
        """
        反归一化边界框坐标

        Args:
            bboxes (list): 归一化后的边界框列表，每个元素为 BaseBbox 实例
            width (int): 图片宽度
            height (int): 图片高度

        Returns:
            list: 反归一化后的边界框列表，每个元素为 BaseBbox 实例
        """
        return [BaseBbox.unnormalize(bbox, width=width, height=height) for bbox in bboxes]


def bbox(
    bboxes: list = None,
    *,
    bbox_type: str = "xmin_ymin_xmax_ymax",
    normalize: bool = True,
    width: int = None,
    height: int = None,
) -> Bbox:
    """
    创建 Bbox 实例

    Args:
        bboxes (list): 边界框列表，每个元素为 BaseBbox 实例
        bbox_type (str, optional): 边界框类型，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"，默认为 "xmin_ymin_xmax_ymax"
        normalize (bool, optional): 是否归一化边界框坐标，默认为 True
        width (int, optional): 图片宽度，默认为 None
        height (int, optional): 图片高度，默认为 None

    Returns:
        Bbox: Bbox 实例

    Raises:
        ValueError: 如果 normalize 为 False 时，width 和 height 未提供
    """
    if normalize:
        return Bbox(bboxes, bbox_type=bbox_type)
    else:
        if width is None or height is None:
            raise ValueError("normalize 为 False 时，width 和 height 必须提供")
        return Bbox(Bbox.normalize(bboxes, width=width, height=height), bbox_type=bbox_type)


def read_txt_label_file(label_file_path: str, bbox_type: str = "center_w_h") -> Bbox:
    """
    读取标签文件并返回边界框实例

    Args:
        label_file_path (str): 标签文件路径
        bbox_type (str, optional): 边界框类型，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"，默认为 "xmin_ymin_xmax_ymax"

    Returns:
        Bbox: 边界框实例
    """
    obj_bbox = bbox(bbox_type=bbox_type)
    with open(label_file_path, "r") as file:
        for line in file.readlines():
            line = line.strip().split()
            line[0] = int(line[0])
            line[1:] = [float(x) for x in line[1:]]
            obj_bbox.append(line)
    return obj_bbox


def save_txt_label_file(label_file_path: str, bbox: Bbox, bbox_type: str = "xmin_ymin_xmax_ymax"):
    """
    将边界框实例保存为标签文件

    Args:
        label_file_path (str): 标签文件路径
        bboxes (Bbox): 边界框实例
        bbox_type (str, optional): 边界框类型，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"，默认为 "xmin_ymin_xmax_ymax"
    """
    with open(label_file_path, "w") as file:
        file.write(str(bbox.convert(bbox_type)))


def mask_to_bbox(mask: np.ndarray, mask_type: str = "gray") -> Bbox:
    """
    将二值掩码转换为边界框

    Args:
        np_mask (np.ndarray): 二值掩码数组
        mask_type (str, optional): 掩码类型，可选值为 "gray"，默认为 "gray"

    Returns:
        Bbox: 边界框实例

    Raises:
        ValueError: 如果 np_mask 不是 2 维数组
        ValueError: 如果 mask_type 不是 'gray'
    """
    if mask.ndim != 2:
        raise ValueError("np_mask 必须是 2 维数组")
    if mask_type == "gray":
        _mask = mask != 0
        (y_indices,) = np.nonzero(np.any(_mask == 1, axis=1))
        (x_indices,) = np.nonzero(np.any(_mask == 1, axis=0))
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
    else:
        raise ValueError("mask_type 必须是 'gray'")
    return bbox([[0, x_min, y_min, x_max, y_max]], normalize=False, width=_mask.shape[1], height=_mask.shape[0])


def extract_class_names_from_xml_file(xml_file_path: str):
    """
    从XML文件中自动提取所有类别名称

    Args:
        xml_file_path (str): XML 文件路径

    Returns:
        list: 排序后的唯一类别名称列表
    """
    if not Path(xml_file_path).exists():
        raise FileNotFoundError(f"文件 {xml_file_path} 不存在")
    if not Path(xml_file_path).suffix == ".xml":
        raise ValueError(f"文件 {xml_file_path} 不是 XML 文件")
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    objects = root.findall("object")
    class_names = set()
    for obj in objects:
        class_names.add(obj.find("name").text)
    return class_names


def extract_class_names_from_xml_dir(xml_dir: str):
    """
    从XML目录中自动提取所有类别名称

    Args:
        xml_dir (str): XML 文件目录

    Returns:
        list: 排序后的唯一类别名称列表
    """
    xml_dir_path = Path(xml_dir)
    class_names = set()
    for xml_file in xml_dir_path.iterdir():
        if xml_file.suffix == ".xml":
            class_names.update(extract_class_names_from_xml_file(xml_file))
    return sorted(list(class_names))


def read_xml_label_file(xml_file_path: str, class_names: list, bbox_type: str = "xmin_ymin_xmax_ymax") -> Bbox:
    """
    读取 XML 文件并返回边界框实例

    Args:
        xml_file_path (str): XML 文件路径
        class_names (list): 类别名称列表
        bbox_type (str, optional): 边界框类型，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"，默认为 "xmin_ymin_xmax_ymax"

    Returns:
        Bbox: 边界框实例
    """
    if not Path(xml_file_path).exists():
        raise FileNotFoundError(f"文件 {xml_file_path} 不存在")
    if not Path(xml_file_path).suffix == ".xml":
        raise ValueError(f"文件 {xml_file_path} 不是 XML 文件")
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find("size")
    width, height = [float(size.find(tag).text) for tag in ["width", "height"]]

    # 提取边界框
    objects = root.findall("object")
    obj_bbox = bbox(bbox_type=bbox_type)
    for obj in objects:
        class_name = obj.find("name").text
        if class_name in class_names:
            class_index = class_names.index(class_name)
            xml_bbox = [float(obj.find("bndbox").find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            obj_bbox.append(BaseBbox.normalize([int(class_index), *xml_bbox], width=width, height=height))
    return obj_bbox


def generate_data_yaml(class_names: list, output_dir: str):
    """
    生成YOLOv8训练所需的data.yaml配置文件

    Args:
        class_names (list): 类别名称列表
        output_dir (str): 输出目录
    """
    data = {
        "path": "./dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(Path(output_dir) / "data.yaml", "w") as file:
        yaml.dump(data, file)
    print(f"已生成YOLOv8配置文件: {Path(output_dir) / 'data.yaml'}")


def batch_xml_to_txt(
    xml_dir: str,
    txt_dir: str,
    read_bbox_type: str = "xmin_ymin_xmax_ymax",
    save_bbox_type: str = "center_w_h",
):
    """
    批量将 XML 文件转换为 YOLO 格式的 TXT 文件

    Args:
        xml_dir (str): XML 文件目录
        txt_dir (str): 输出 TXT 文件目录
    """
    if not Path(xml_dir).exists():
        raise FileNotFoundError(f"目录 {xml_dir} 不存在")
    if not Path(xml_dir).is_dir():
        raise ValueError(f"路径 {xml_dir} 不是目录")
    Path(txt_dir).mkdir(parents=True, exist_ok=True)
    class_names = extract_class_names_from_xml_dir(xml_dir)
    for xml_file in Path(xml_dir).iterdir():
        if xml_file.suffix == ".xml":
            txt_file = Path(txt_dir) / (xml_file.stem + ".txt")
            bbox = read_xml_label_file(str(xml_file), class_names, read_bbox_type)
            save_txt_label_file(str(txt_file), bbox, save_bbox_type)
    print(f"转换完成，已转换 {len(list(Path(txt_dir).iterdir()))} 个文件")
    generate_data_yaml(class_names, txt_dir)


"""
文件重命名
"""


def rename_file(file_path: str, new_name: str):
    """
    重命名文件

    Args:
        file_path (str): 文件路径
        new_name (str): 新文件名

    Raises:
        FileExistsError: 如果新文件名已存在
    """
    _file_path = Path(file_path)
    file_new_name = _file_path.name.replace(_file_path.stem, new_name)
    file_new_path = _file_path.parent / file_new_name
    if file_new_path.exists():
        raise FileExistsError(f"文件 {file_new_path} 已存在")
    _file_path.rename(file_new_path)


def batch_rename(image_dir_path: str, label_dir_path: str, *, prefix: str, offset: int = 0):
    """
    批量重命名图片和标签文件

    Args:
        image_dir_path (str): 图片目录路径
        label_dir_path (str): 标签目录路径
        prefix (str): 文件名前缀
        offset (int, optional): 文件名偏移量，默认为 0
    """
    print(f"以 {prefix} 为前缀重命名文件")
    _image_dir_path, _label_dir_path = Path(image_dir_path), Path(label_dir_path)
    for index, image_path in enumerate(_image_dir_path.iterdir()):
        label_path = _label_dir_path / (image_path.stem + ".txt")
        try:
            rename_file(str(image_path), f"{prefix}_{index + offset:010d}")
            rename_file(str(label_path), f"{prefix}_{index + offset:010d}")
        except FileExistsError as e:
            print(e)
            batch_rename(image_dir_path, label_dir_path, prefix="temp")
            batch_rename(image_dir_path, label_dir_path, prefix=prefix)
            break
    print("重命名完成")
