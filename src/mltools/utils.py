import json
import time
import subprocess
from pathlib import Path
from nvitop import Device


class DataSaveToJson:
    """
    json数据保存器，提供将数据保存到 JSON 文件和从 JSON 文件加载数据的功能。
    """

    @staticmethod
    def save_data(path: str, label: str, datas: dict):
        """
        保存数据到指定路径的 JSON 文件中。

        Args:
            path (str): JSON 文件的保存路径。
            label (str): 数据在 JSON 文件中的键名。
            datas (dict): 要保存的数据。
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
    def load_data(path: str, label: str) -> any:
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


class Accumulator:
    """
    在 n 个变量上累加，用于统计训练过程中的指标。
    """

    def __init__(self, n: int):
        """
        初始化累加器。

        Args:
            n (int): 变量个数。
        """
        self.data = [0.0] * n

    def add(self, *args: int | float):
        """
        添加数据到累加器。

        Args:
            *args (int | float): 要添加的数据。
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """
        重置累加器的数据。
        """
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
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

    def __init__(self, n: int):
        """
        初始化记录器。

        Args:
            n (int): 记录器的数量。
        """
        self.data = [[] for _ in range(n)]

    def get_latest_record(self) -> list[float]:
        """
        返回最新记录。

        Returns:
            list[float]: 最新记录的列表。
        """
        return [item[-1] for item in self.data]

    def max_record_size(self) -> int:
        """
        返回最长记录长度。

        Returns:
            int: 最长记录的长度。
        """
        return max([len(item) for item in self.data])

    def reset(self):
        """
        重置记录器的数据。
        """
        self.data = [[] for _ in range(len(self.data))]

    def __getitem__(self, idx: int) -> list[float]:
        """
        返回第 n 个记录器的数据。

        Args:
            idx (int): 索引。

        Returns:
            list[float]: 第 idx 个记录器的数据列表。
        """
        return self.data[idx]

    def save(self, path: str, label: str = "recorder"):
        """
        保存记录器的数据到 JSON 文件。

        Args:
            path (str): JSON 文件的保存路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'recorder'。
        """
        DataSaveToJson.save_data(path, label, self.data)

    def load(self, path: str, label: str = "recorder"):
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

    def stop(self) -> float:
        """
        停止计时器并将时间记录在列表中。

        Returns:
            float: 本次记录的时间。
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self) -> float:
        """
        返回平均时间。

        Returns:
            float: 平均时间，单位为秒。如果没有记录时间，则返回 0。
        """
        if self.times:
            return sum(self.times) / len(self.times)
        else:
            return 0

    def sum(self) -> float:
        """
        计算记录的所有时间的总和。

        Returns:
            float: 记录的所有时间的总和，单位为秒。如果没有记录时间，则返回 0。
        """
        return sum(self.times)

    @staticmethod
    def str(times: float) -> str:
        """
        将时间转换为格式化的字符串。

        Args:
            times (float): 时间，单位为秒。

        Returns:
            str: 格式化后的时间字符串，格式为 "HH:MM:SS"。
        """
        return time.strftime("%H:%M:%S", time.gmtime(times))

    def save(self, path: str, label: str = "timer"):
        """
        保存计时器的时间数据到 JSON 文件。

        Args:
            path (str): JSON 文件的保存路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'timer'。
        """
        DataSaveToJson.save_data(path, label, self.times)

    def load(self, path: str, label: str = "timer"):
        """
        从 JSON 文件中加载计时器的时间数据。

        Args:
            path (str): JSON 文件的路径。
            label (str, optional): 数据在 JSON 文件中的键名。默认值为 'timer'。
        """
        self.times = DataSaveToJson.load_data(path, label)


def add_ignore_file(dir: str):
    """
    为指定目录添加 .gitignore 文件，用于忽略所有文件。

    Args:
        dir (str): 目录路径。
    """
    file = Path(dir) / ".gitignore"
    if not file.exists():
        with open(file, "w") as f:
            f.write("*\n")


def bash_command(command: str | list[str]):
    """
    执行 bash 命令并实时打印输出。

    Args:
        command (str | list[str]): 要执行的 bash 命令。
    """
    if isinstance(command, str):
        command = command.split()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 合并错误输出到标准输出
        text=True,  # 返回字符串而非字节
        bufsize=1,  # 行缓冲模式
    )
    while True:
        line = process.stdout.readline()
        if not line:
            break  # 进程结束
        print(line.strip())  # 实时打印


def get_gpu(func):
    """
    装饰器，用于获取空闲的 GPU 设备并执行指定函数。

    Args:
        func (callable): 要在 GPU 上执行的函数。
    """
    # 获取所有可用的GPU设备
    devices = Device.all()
    searching_for_gpu = True

    while searching_for_gpu:
        for device in devices:
            processes = device.processes()
            print("-------")
            print(device.index, len(processes.items()))

            if len(processes.items()) == 1:
                searching_for_gpu = False
                print("-------")

                def wrapper(*args: list[str], **kwargs: dict[str, str]):
                    func(*args, device=f"cuda:{device.index}", **kwargs)

                break
            time.sleep(2)
    return wrapper
