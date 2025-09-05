import torch
from collections import Counter
from mltools import utils


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
