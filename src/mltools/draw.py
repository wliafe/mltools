import numpy as np
from IPython import display
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import patches as patches
from mltools import data as ma


def set_axes(axes: matplotlib.axes.Axes | list[matplotlib.axes.Axes], *, axis: bool = True, **kwargs: dict):
    """
    设置axes。

    Args:
        axes (matplotlib.axes.Axes | list[matplotlib.axes.Axes]): 子图对象列表。
        axis (bool, optional): 是否显示坐标轴。默认值为True。
        **kwargs (dict): 其他axes设置参数。
    """
    axes_list = axes
    if isinstance(axes_list, matplotlib.axes.Axes):
        axes_list = [axes_list]
    if isinstance(axes_list[0], list):
        axes_list = [ax for ax_list in axes_list for ax in ax_list]
    for ax in axes_list:
        if not axis:
            ax.set_axis_off()
        ax.set(**kwargs)


class Animator:
    """
    在动画中绘制数据，用于动态展示训练过程中的指标变化。
    """

    def __init__(
        self,
        xlabel: str = None,
        ylabel: str = None,
        xlim: tuple[int, int] = None,
        ylim: tuple[int, int] = None,
        legend: list[str] = None,
        fmts: list[str] = None,
    ):
        """
        初始化动画器。

        Args:
            xlabel (str, optional): x轴标签。默认值为None。
            ylabel (str, optional): y轴标签。默认值为None。
            xlim (tuple[int, int], optional): x轴范围。默认值为None。
            ylim (tuple[int, int], optional): y轴范围。默认值为None。
            legend (list[str], optional): 图例。默认值为None。
            fmts (list[str], optional): 线条格式。默认值为None。
        """
        self.fig, self.ax = plt.subplots()  # 生成画布
        self.set_axes = lambda: set_axes(
            self.ax, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim
        )  # 初始化设置axes函数
        self.legend = legend  # 图例
        self.fmts = fmts if fmts else ("-", "m--", "g-.", "r:")  # 格式
        plt.close()

    def show(self, Y: list[list[float]]):
        """
        展示动画。

        Args:
            Y (list[list[float]]): y轴数据列表。
        """
        X = [list(range(1, len(sublist) + 1)) for sublist in Y]
        self.ax.cla()  # 清除画布
        for x, y, fmt in zip(X, Y, self.fmts):
            self.ax.plot(x, y, fmt)
        self.set_axes()  # 设置axes
        if self.legend:
            self.ax.legend(self.legend)  # 设置图例
        self.ax.grid()  # 设置网格线
        display.display(self.fig)  # 画图
        display.clear_output(wait=True)  # 清除输出

    def save(self, path: str):
        """
        保存动画为图片文件。

        Args:
            path (str): 图片文件的保存路径。
        """
        self.fig.savefig(path)


def images(images: np.ndarray, labels: list[str], shape: tuple[int, int]):
    """
    展示图片。

    Args:
        images (np.ndarray): 图片数据数组。
        labels (list[str]): 图片标签列表。
        shape (tuple[int, int]): 子图布局形状。

    Raises:
        TypeError: 如果images不是numpy数组。
    """
    if not isinstance(images, np.ndarray):
        raise TypeError("images must be a numpy array")
    fig, axes = plt.subplots(*shape)
    axes = [element for sublist in axes for element in sublist]
    for ax, img, label in zip(axes, images, labels):
        set_axes(ax, axis=False, title=label)
        ax.imshow(img, cmap="gray")
    plt.show()


def numpy_to_image(numpy_array: np.ndarray):
    """
    展示图片。

    Args:
        numpy_array (np.ndarray): 图片数据数组。

    Raises:
        TypeError: 如果numpy_array不是numpy数组。
    """
    if not isinstance(numpy_array, np.ndarray):
        raise TypeError("numpy_array must be a numpy array")
    fig, ax = plt.subplots(1, 1)
    if numpy_array.ndim == 2:
        ax.imshow(numpy_array, cmap="gray")  # 使用灰度图
    elif numpy_array.ndim == 3:
        ax.imshow(numpy_array)
    set_axes(ax, axis=False)
    plt.show()  # 显示图片


def draw_bbox(image_path: str, bbox: ma.Bbox):
    """
    绘制边界框。

    Args:
        image_path (str): 图片文件路径。
        bbox (data.Bbox): 边界框对象。
    """
    image = mpimg.imread(image_path)
    rect_bboxes = ma.Bbox.unnormalize(bbox.xmin_ymin_w_h().to_list(), width=image.shape[1], height=image.shape[0])
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)  # 显示原图
    axes[1].imshow(image)  # 显示矩形框图
    for rect_bbox in rect_bboxes:
        rect = patches.Rectangle(
            (rect_bbox[1], rect_bbox[2]), rect_bbox[3], rect_bbox[4], linewidth=2, edgecolor="r", facecolor="none"
        )  # 创建矩形框
        axes[1].add_patch(rect)  # 将矩形框添加到坐标轴
    set_axes(axes, axis=False)
    plt.show()
