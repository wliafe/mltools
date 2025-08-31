from torch.utils import data
import re
import httpx
import numpy as np
from PIL import Image
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


class BaseBbox:
    def __init__(self, bbox: list, *, bbox_type: str = "xmin_ymin_xmax_ymax"):
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

    def __str__(self):
        return f"{self.class_id} {self.x_min} {self.y_min} {self.x_max} {self.y_max}\n"

    def __repr__(self):
        return f"BaseBbox(class_id={self.class_id}, bbox=[{self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}])"

    def xmin_ymin_xmax_ymax(self):
        return [self.class_id, self.x_min, self.y_min, self.x_max, self.y_max]

    def xmin_ymin_w_h(self):
        return [self.class_id, self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min]

    def center_w_h(self):
        return [
            self.class_id,
            self.x_min + (self.x_max - self.x_min) / 2,
            self.y_min + (self.y_max - self.y_min) / 2,
            self.x_max - self.x_min,
            self.y_max - self.y_min,
        ]

    @staticmethod
    def normalize(bbox: list, *, width: int, height: int):
        bbox[1] = bbox[1] / width
        bbox[2] = bbox[2] / height
        bbox[3] = bbox[3] / width
        bbox[4] = bbox[4] / height
        return bbox

    @staticmethod
    def unnormalize(bbox: list, *, width: int, height: int):
        return [bbox[0], bbox[1] * width, bbox[2] * height, bbox[3] * width, bbox[4] * height]


class Bbox:
    def __init__(self, bboxes: list, *, bbox_type: str = "xmin_ymin_xmax_ymax"):
        if not isinstance(bboxes, list):
            raise ValueError("bboxes 参数必须是列表")
        if not all(isinstance(bbox, list) for bbox in bboxes):
            raise ValueError("bboxes 列表元素必须是列表")
        self.bboxes = [BaseBbox(bbox, bbox_type=bbox_type) for bbox in bboxes]

    def __getitem__(self, index: int):
        return self.bboxes[index]

    def __len__(self):
        return len(self.bboxes)

    def __str__(self):
        return "".join([str(bbox) for bbox in self.bboxes])

    def __repr__(self):
        return f"Bbox({self.bboxes})"

    def xmin_ymin_xmax_ymax(self):
        return [bbox.xmin_ymin_xmax_ymax() for bbox in self.bboxes]

    def xmin_ymin_w_h(self):
        return [bbox.xmin_ymin_w_h() for bbox in self.bboxes]

    def center_w_h(self):
        return [bbox.center_w_h() for bbox in self.bboxes]

    @staticmethod
    def normalize(bboxes: list, *, width: int, height: int):
        return [BaseBbox.normalize(bbox, width=width, height=height) for bbox in bboxes]

    @staticmethod
    def unnormalize(bboxes: list, *, width: int, height: int):
        return [BaseBbox.unnormalize(bbox, width=width, height=height) for bbox in bboxes]


def bbox(
    bboxes: list,
    *,
    bbox_type: str = "xmin_ymin_xmax_ymax",
    normalize: bool = True,
    width: int = None,
    height: int = None,
):
    if normalize:
        return Bbox(bboxes, bbox_type=bbox_type)
    else:
        if width is None or height is None:
            raise ValueError("normalize 为 False 时，width 和 height 必须提供")
        return Bbox(Bbox.normalize(bboxes, width=width, height=height), bbox_type=bbox_type)


def mask_to_bbox(np_mask: np.ndarray, mask_type: str = "gray"):
    if np_mask.ndim != 2:
        raise ValueError("np_mask 必须是 2 维数组")
    if mask_type == "gray":
        mask = np_mask != 0
        (y_indices,) = np.nonzero(np.any(mask == 1, axis=1))
        (x_indices,) = np.nonzero(np.any(mask == 1, axis=0))
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
    else:
        raise ValueError("mask_type 必须是 'gray'")
    return bbox([[0, x_min, y_min, x_max, y_max]], normalize=False, width=np_mask.shape[1], height=np_mask.shape[0])


def masks_to_bbox(data_path: str):
    path = Path(data_path)
    mask_path = path / "masks"
    bbox_path = path / "labels"
    bbox_path.mkdir(parents=True, exist_ok=True)

    for mask in mask_path.iterdir():
        image = Image.open(mask)
        np_image = np.max(np.array(image), axis=2)
        bboxes = mask_to_bbox(np_image).center_w_h()
        with open(bbox_path / (mask.stem + ".txt"), "w") as file:
            file.write(bboxes)


def read_label_file(label_file_path: str):
    with open(label_file_path, "r") as file:
        lines = []
        for line in file.readlines():
            line = line.strip().split()
            line[0] = int(line[0])
            line[1:] = [float(x) for x in line[1:]]
            lines.append(line)
        return lines


def batch_rename(train_image_path: str, train_label_path: str, *, prefix: str, offset: int = 0):
    train_image_path, train_label_path = Path(train_image_path), Path(train_label_path)
    for index, image_item in enumerate(train_image_path.iterdir()):
        label_item = train_label_path / (image_item.stem + ".txt")
        if image_item.stem == label_item.stem:
            new_name = image_item.name.replace(image_item.stem, f"{prefix}_{index + offset:010d}")
            new_path = image_item.parent / new_name
            image_item.rename(new_path)
            new_name = label_item.name.replace(label_item.stem, f"{prefix}_{index + offset:010d}")
            new_path = label_item.parent / new_name
            label_item.rename(new_path)
