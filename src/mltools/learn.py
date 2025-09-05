import torch
from torch import nn
import logging
from pathlib import Path
from datetime import datetime
from mltools import utils, draw


class Epoch:
    """
    机器学习 Epoch，用于管理训练轮数，支持保存和加载总训练轮数。
    """

    def __init__(self, parent: object):
        """
        初始化

        Args:
            parent (object): 父对象，用于访问日志记录器。
        """
        self._totol_epoch = 0
        self.parent = parent

    def __call__(self, num_epochs: int) -> int:
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
            self.parent.logger.warning(
                f"num_epochs is {num_epochs}, less than totol training epoch {self.totol_epoch}, the model won't be trained."
            )
        return num_epoch

    @property
    def totol_epoch(self) -> int:
        """
        返回总迭代次数。

        Returns:
            int: 总训练轮数。
        """
        return self._totol_epoch

    def save(self, path: str, label: str = "epoch"):
        """
        保存总训练轮数到 JSON 文件。

        Args:
            path (str): JSON 文件的保存路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'epoch'。
        """
        utils.DataSaveToJson.save_data(path, label, self.totol_epoch)

    def load(self, path: str, label: str = "epoch"):
        """
        从 JSON 文件中加载总训练轮数。

        Args:
            path (str): JSON 文件的路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'epoch'。
        """
        self._totol_epoch = utils.DataSaveToJson.load_data(path, label)


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

    def add_save_func(self, func: callable):
        """
        添加保存函数。

        Args:
            func (callable): 保存函数。
        """
        self.save_func.append(func)

    def save(self, dir_path: str):
        """
        保存数据。

        Args:
            dir_path (str): 数据保存的目录路径。
        """
        for func in self.save_func:
            func(dir_path)

    def add_load_func(self, func: callable):
        """
        添加加载函数。

        Args:
            func (callable): 加载函数。
        """
        self.load_func.append(func)

    def load(self, dir_path: str):
        """
        加载数据。

        Args:
            dir_path (str): 数据加载的目录路径。
        """
        for func in self.load_func:
            func(dir_path)


class MachineLearning:
    """
    机器学习工具类，提供批量创建训练辅助对象、管理模型和数据的保存与加载等功能。
    """

    def __init__(self, file_name: str):
        """
        初始化函数。

        Args:
            file_name (str): 文件名。
        """
        # 创建目录
        Path("../data").mkdir(parents=True, exist_ok=True)
        utils.add_ignore_file("../data")
        Path("../results").mkdir(parents=True, exist_ok=True)
        utils.add_ignore_file("../results")

        # 定义时间字符串和文件名
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.dir_path = f"../results/{time_str}-{file_name}"
        self.file_name = file_name

        # 创建目录
        Path(self.dir_path).mkdir()

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

    def batch_create(self, create_epoch: bool = True, create_timer: bool = True, create_recorder: bool = True) -> tuple:
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

    def save(self, dir_name: str = None):
        """
        保存数据。

        Args:
            dir_name (str, optional): 数据保存的目录名。默认值为 None。
        """
        dir_path = f"../results/{dir_name}" if dir_name else self.dir_path
        self.data_manager.save(dir_path)

    def load(self, dir_name: str = None):
        """
        加载数据。

        Args:
            dir_name (str, optional): 数据加载的目录名。默认值为 None。
        """
        dir_path = f"../results/{dir_name}" if dir_name else self.dir_path
        self.data_manager.load(dir_path)

    def create_epoch(self, label: str = "num_epochs") -> Epoch:
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

    def create_timer(self, label: str = "timer") -> utils.Timer:
        """
        创建计时器。

        Args:
            label (str, optional): 计时器的标签，建议和被赋值变量名相同。默认值为 'timer'。

        Returns:
            Timer: 创建的计时器对象。
        """
        timer = utils.Timer()

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

    def create_recorder(self, recorder_num: int, label: str = "recorder") -> utils.Recorder:
        """
        创建记录器。

        Args:
            recorder_num (int): 记录器的数量。
            label (str, optional): 记录器的标签，建议和被赋值变量名相同。默认值为 'recorder'。

        Returns:
            Recorder: 创建的记录器对象。
        """
        recorder = utils.Recorder(recorder_num)

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

    def create_animator(
        self,
        xlabel: str = None,
        ylabel: str = None,
        xlim: tuple = None,
        ylim: tuple = None,
        legend: list = None,
        fmts: list = None,
        label: str = "animator",
    ) -> draw.Animator:
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
        animator = draw.Animator(xlabel, ylabel, xlim, ylim, legend, fmts)

        def save(dir_path):
            animator.save(f"{dir_path}/{self.file_name}.png")
            self.logger.debug(f"save Animator({label}) to {dir_path}/{self.file_name}.png")

        self.data_manager.add_save_func(save)

        self.logger.debug(f"create Animator({label})")
        return animator

    def add_model(self, model: nn.Module, label: str = "model"):
        """
        添加模型。

        Args:
            model: 模型。
            label (str, optional): 模型的标签，建议和模型变量名相同。默认值为 'model'。

        Raises:
            RuntimeError: 如果模型不是 nn.Module 类型。
        """
        if not isinstance(model, nn.Module):
            raise RuntimeError(f"model({label}) must be a nn.Module")

        def save(dir_path):
            torch.save(model.state_dict(), f"{dir_path}/{self.file_name}.pt")
            self.logger.debug(f"save model({label}) to {dir_path}/{self.file_name}.pt")

        self.data_manager.add_save_func(save)

        def load(dir_path):
            model.load_state_dict(torch.load(f"{dir_path}/{self.file_name}.pt"))
            self.logger.debug(f"load model({label}) from {dir_path}/{self.file_name}.pt")

        self.data_manager.add_load_func(load)

        self.logger.debug(f"add model({label})")
        self.logger.debug(f"model({label}) is {model}")

    def print_training_time_massage(self, timer: utils.Timer, num_epochs: int, current_epoch: int):
        """
        打印模型训练时间相关信息，包括已训练时长、平均训练时长和预估剩余训练时长。

        Args:
            timer (Timer): 计时器对象，用于获取训练时间数据。
            num_epochs (int): 总训练轮数。
            current_epoch (int): 当前训练到的轮数。
        """
        # 计算已训练的总时长，并转换为 HH:MM:SS 格式
        trained_duration = utils.Timer.str(timer.sum())
        # 计算每轮的平均训练时长，并转换为 HH:MM:SS 格式
        average_duration = utils.Timer.str(timer.avg())
        # 计算预估的剩余训练时长，并转换为 HH:MM:SS 格式
        estimated_duration = utils.Timer.str((num_epochs - current_epoch) * timer.avg())
        # 打印训练时间相关信息
        self.logger.info(
            f"Trained duration: {trained_duration}, Average training duration: {average_duration}, Estimated training duration:{estimated_duration}"
        )

    def model_params(self, model: nn.Module, label: str = "model"):
        """
        打印模型参数数量。

        Args:
            model: 模型对象。

        Raises:
            RuntimeError: 如果模型不是 nn.Module 类型。
        """
        if not isinstance(model, nn.Module):
            raise RuntimeError(f"model({label}) must be a nn.Module")
        # 统计模型参数数量
        num_params = sum([param.numel() for param in model.parameters()])
        # 打印模型参数数量
        self.logger.info(f"Number of model({label}) parameters: {num_params / (1000 * 1000):.2f}M")
