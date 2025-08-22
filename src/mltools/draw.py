from IPython import display
from matplotlib import pyplot as plt


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
