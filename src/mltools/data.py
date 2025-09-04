from torch.utils import data
import re
import httpx
import numpy as np
from tqdm import tqdm
from pathlib import Path


class MyDataset(data.Dataset):
    """
    自定义数据集类，继承自 torch.utils.data.Dataset，用于管理机器学习任务中的数据。

    功能：
    - 存储和管理数据集
    - 支持索引访问数据
    - 提供数据集长度信息
    """

    def __init__(self, datas: list):
        """
        初始化数据集

        参数:
            datas (list): 数据集内容，可以是任何格式的数据
        """
        data.Dataset.__init__(self)
        self.data = datas

    def __len__(self) -> int:
        """
        返回数据集的样本数量

        返回:
            int: 数据集中的样本总数
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> any:
        """
        根据索引获取单个数据样本

        参数:
            idx (int): 数据索引

        返回:
            对应索引的数据样本
        """
        return self.data[idx]


def split_data(datas: list, ratio: list) -> list:
    """
    划分数据集

    参数:
        datas (list): 数据集内容，可以是任何格式的数据
        ratio (list): 划分比例，例如 [0.8, 0.2] 表示划分成 80% 训练集和 20% 测试集

    返回:
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

    参数:
        datas (list): 数据集内容，可以是任何格式的数据
        batch_size (int): 每个批次的样本数量
        shuffle (bool, optional): 是否在每个 epoch 开始时打乱数据. 默认值为 True.
        num_workers (int, optional): 用于数据加载的子进程数量. 默认值为 0.
        pin_memory (bool, optional): 是否将数据加载到 CUDA 固定内存中. 默认值为 False.
        drop_last (bool, optional): 是否丢弃最后一个批次, 如果数据集大小不能被批次大小整除. 默认值为 False.

    返回:
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


def download_file(url: str, *, save_path: str) -> str:
    """
    下载文件

    参数:
        url (str): 文件的 URL 地址
        save_path (str): 保存文件的路径

    返回:
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


class BaseBbox:
    """
    基础边界框类，用于表示物体的边界框。
    功能:
        表示物体的边界框，包含类别 ID 和坐标信息。
    属性:
        class_id (int): 物体的类别 ID。
        x_min (float): 边界框的左边界。
        y_min (float): 边界框的上边界。
        x_max (float): 边界框的右边界。
        y_max (float): 边界框的下边界。
    """

    def __init__(self, bbox: list, *, bbox_type: str = "xmin_ymin_xmax_ymax"):
        """
        初始化边界框

        参数:
            bbox (list): 边界框参数，格式根据 bbox_type 不同而不同
            bbox_type (str, optional): 边界框格式，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h"、"center_w_h". 默认值为 "xmin_ymin_xmax_ymax".

        抛出:
            ValueError: 如果 bbox 参数长度不是 5 个元素
            ValueError: 如果类别不是整数
            ValueError: 如果 bbox 参数不归一化
            ValueError: 如果 bbox_type 不是 'xmin_ymin_xmax_ymax'、'xmin_ymin_w_h' 或 'center_w_h'
        """
        if len(bbox) != 5:
            raise ValueError("bbox 参数必须是 5 个元素")
        if isinstance(bbox[0], int):
            self.class_id = bbox[0]
        else:
            raise ValueError("类别必须是整数")
        if not all(isinstance(item, float) for item in bbox[1:]):
            raise ValueError("bbox 参数必须归一化")
        if bbox_type == "xmin_ymin_xmax_ymax":
            self.x_min, self.y_min, self.x_max, self.y_max = bbox[1:]
        elif bbox_type == "xmin_ymin_w_h":
            self.x_min, self.y_min, self.x_max, self.y_max = (bbox[1], bbox[2], bbox[1] + bbox[3], bbox[2] + bbox[4])
        elif bbox_type == "center_w_h":
            self.x_min, self.y_min, self.x_max, self.y_max = (
                bbox[1] - bbox[3] / 2,
                bbox[2] - bbox[4] / 2,
                bbox[1] + bbox[3] / 2,
                bbox[2] + bbox[4] / 2,
            )
        else:
            raise ValueError("bbox_type 必须是 'xmin_ymin_xmax_ymax'、'xmin_ymin_w_h' 或 'center_w_h'")

    def __str__(self) -> str:
        """
        返回边界框的字符串表示

        返回:
            str: 边界框的字符串表示，格式为 "class_id x_min y_min x_max y_max"
        """
        return f"{self.class_id} {self.x_min} {self.y_min} {self.x_max} {self.y_max}"

    def __repr__(self) -> str:
        """
        返回边界框的字符串表示

        返回:
            str: 边界框的字符串表示，格式为 "BaseBbox(class_id={self.class_id}, bbox=[{self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}])"
        """
        return f"BaseBbox(class_id={self.class_id}, bbox=[{self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}])"

    def xmin_ymin_xmax_ymax(self) -> list:
        """
        返回边界框的坐标表示

        返回:
            list: 边界框的坐标表示，格式为 [class_id, x_min, y_min, x_max, y_max]
        """
        return [self.class_id, self.x_min, self.y_min, self.x_max, self.y_max]

    def xmin_ymin_w_h(self) -> list:
        """
        返回边界框的坐标表示

        返回:
            list: 边界框的坐标表示，格式为 [class_id, x_min, y_min, x_max - x_min, y_max - y_min]
        """
        return [self.class_id, self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min]

    def center_w_h(self) -> list:
        """
        返回边界框的坐标表示

        返回:
            list: 边界框的坐标表示，格式为 [class_id, x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2, x_max - x_min, y_max - y_min]
        """
        return [
            self.class_id,
            self.x_min + (self.x_max - self.x_min) / 2,
            self.y_min + (self.y_max - self.y_min) / 2,
            self.x_max - self.x_min,
            self.y_max - self.y_min,
        ]

    @staticmethod
    def normalize(bbox: list, *, width: int, height: int) -> list:
        """
        归一化边界框坐标

        参数:
            bbox (list): 边界框参数，格式为 [class_id, x_min, y_min, x_max, y_max]
            width (int): 图片宽度
            height (int): 图片高度

        返回:
            list: 归一化后的边界框参数，格式为 [class_id, x_min / width, y_min / height, x_max / width, y_max / height]
        """
        return [bbox[0], bbox[1] / width, bbox[2] / height, bbox[3] / width, bbox[4] / height]

    @staticmethod
    def unnormalize(bbox: list, *, width: int, height: int) -> list:
        """
        反归一化边界框坐标

        参数:
            bbox (list): 归一化后的边界框参数，格式为 [class_id, x_min / width, y_min / height, x_max / width, y_max / height]
            width (int): 图片宽度
            height (int): 图片高度

        返回:
            list: 反归一化后的边界框参数，格式为 [class_id, int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)]
        """
        return [bbox[0], int(bbox[1] * width), int(bbox[2] * height), int(bbox[3] * width), int(bbox[4] * height)]


class Bbox:
    """
    边界框容器类

    属性:
        bboxes (list): 边界框列表，每个元素为 BaseBbox 实例
    """

    def __init__(self, bboxes: list, *, bbox_type: str = "xmin_ymin_xmax_ymax"):
        """
        初始化 Bbox 实例

        参数:
            bboxes (list): 边界框列表，每个元素为 BaseBbox 实例
            bbox_type (str, optional): 边界框类型，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"，默认为 "xmin_ymin_xmax_ymax"
        抛出:
            ValueError: 如果 bboxes 参数不是列表
            ValueError: 如果 bboxes 列表元素不是列表
        """
        if not isinstance(bboxes, list):
            raise ValueError("bboxes 参数必须是列表")
        if not all(isinstance(bbox, list) for bbox in bboxes):
            raise ValueError("bboxes 列表元素必须是列表")
        self.bboxes = [BaseBbox(bbox, bbox_type=bbox_type) for bbox in bboxes]

    def __getitem__(self, index: int) -> BaseBbox:
        """
        获取指定索引的边界框

        参数:
            index (int): 边界框索引

        返回:
            BaseBbox: 指定索引的边界框实例
        """
        return self.bboxes[index]

    def __len__(self) -> int:
        """
        返回边界框列表的长度

        返回:
            int: 边界框列表的长度
        """
        return len(self.bboxes)

    def __str__(self) -> str:
        """
        返回边界框列表的字符串表示

        返回:
            str: 边界框列表的字符串表示
        """
        return "\n".join(str(bbox) for bbox in self.bboxes)

    def __repr__(self) -> str:
        """
        返回边界框列表的字符串表示

        返回:
            str: 边界框列表的字符串表示
        """
        return "Bbox([\n" + ",\n".join(str(bbox.__repr__()) for bbox in self.bboxes) + ",\n])"

    def xmin_ymin_xmax_ymax(self) -> list:
        """
        返回边界框的坐标表示

        返回:
            list: 边界框的坐标表示，格式为 [class_id, x_min, y_min, x_max, y_max]
        """
        return [bbox.xmin_ymin_xmax_ymax() for bbox in self.bboxes]

    def xmin_ymin_w_h(self) -> list:
        """
        返回边界框的坐标表示

        返回:
            list: 边界框的坐标表示，格式为 [class_id, x_min, y_min, x_max - x_min, y_max - y_min]
        """
        return [bbox.xmin_ymin_w_h() for bbox in self.bboxes]

    def center_w_h(self) -> list:
        """
        返回边界框的坐标表示

        返回:
            list: 边界框的坐标表示，格式为 [class_id, x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2, x_max - x_min, y_max - y_min]
        """
        return [bbox.center_w_h() for bbox in self.bboxes]

    @staticmethod
    def normalize(bboxes: list, *, width: int, height: int) -> list:
        """
        归一化边界框坐标

        参数:
            bboxes (list): 边界框列表，每个元素为 BaseBbox 实例
            width (int): 图片宽度
            height (int): 图片高度

        返回:
            list: 归一化后的边界框列表，每个元素为 BaseBbox 实例
        """
        return [BaseBbox.normalize(bbox, width=width, height=height) for bbox in bboxes]

    @staticmethod
    def unnormalize(bboxes: list, *, width: int, height: int) -> list:
        """
        反归一化边界框坐标

        参数:
            bboxes (list): 归一化后的边界框列表，每个元素为 BaseBbox 实例
            width (int): 图片宽度
            height (int): 图片高度

        返回:
            list: 反归一化后的边界框列表，每个元素为 BaseBbox 实例
        """
        return [BaseBbox.unnormalize(bbox, width=width, height=height) for bbox in bboxes]


def bbox(
    bboxes: list,
    *,
    bbox_type: str = "xmin_ymin_xmax_ymax",
    normalize: bool = True,
    width: int = None,
    height: int = None,
) -> Bbox:
    """
    创建 Bbox 实例

    参数:
        bboxes (list): 边界框列表，每个元素为 BaseBbox 实例
        bbox_type (str, optional): 边界框类型，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"，默认为 "xmin_ymin_xmax_ymax"
        normalize (bool, optional): 是否归一化边界框坐标，默认为 True
        width (int, optional): 图片宽度，默认为 None
        height (int, optional): 图片高度，默认为 None

    返回:
        Bbox: Bbox 实例

    抛出:
        ValueError: 如果 normalize 为 False 时，width 和 height 未提供
    """
    if normalize:
        return Bbox(bboxes, bbox_type=bbox_type)
    else:
        if width is None or height is None:
            raise ValueError("normalize 为 False 时，width 和 height 必须提供")
        return Bbox(Bbox.normalize(bboxes, width=width, height=height), bbox_type=bbox_type)


def read_label_file(label_file_path: str, bbox_type: str = "xmin_ymin_xmax_ymax") -> Bbox:
    """
    读取标签文件并返回边界框实例

    参数:
        label_file_path (str): 标签文件路径
        bbox_type (str, optional): 边界框类型，可选值为 "xmin_ymin_xmax_ymax"、"xmin_ymin_w_h" 或 "center_w_h"，默认为 "xmin_ymin_xmax_ymax"

    返回:
        Bbox: 边界框实例
    """
    lines = []
    with open(label_file_path, "r") as file:
        for line in file.readlines():
            line = line.strip().split()
            line[0] = int(line[0])
            line[1:] = [float(x) for x in line[1:]]
            lines.append(line)
    return bbox(lines, bbox_type=bbox_type)


def mask_to_bbox(mask: np.ndarray, mask_type: str = "gray") -> Bbox:
    """
    将二值掩码转换为边界框

    参数:
        np_mask (np.ndarray): 二值掩码数组
        mask_type (str, optional): 掩码类型，可选值为 "gray"，默认为 "gray"

    返回:
        Bbox: 边界框实例

    抛出:
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


def rename_file(file_path: str, new_name: str):
    """
    重命名文件

    参数:
        file_path (str): 文件路径
        new_name (str): 新文件名

    抛出:
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

    参数:
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
