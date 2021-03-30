from abc import ABCMeta, abstractmethod


class AbstractModel(metaclass=ABCMeta):
    """
    训练模型基类，定义一些通用接口，便于集中控制。注意不要在构造函数中添加过于复杂的逻辑
    """
    @abstractmethod
    def model_type(self):
        """
        返回model所属的大类，比如：分类、回归、图像处理、文本等。
        """
        pass

    @abstractmethod
    def model_name(self):
        """
        返回模块的名字，必须唯一否则会报错，建议言简意赅
        """
        pass

    @abstractmethod
    def run(self):
        """
        开始运行
        """
