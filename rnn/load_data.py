from matplotlib import pyplot as plt
import numpy as np

"""
RNN网络智能处理数值，而非字符串，所以载入输入的基本功能就是将字符进行编码，然后生成一个二维矩阵。矩阵每一行对应文本names文件中的一行，行中的
每一列是一个整型值，代表对应的字符。所以基本过程包含两个步骤：
- 将文本文件中的字符编码
- 将文本内容转化为编码后的矩阵
"""


def load_data():
    """
    从本地names文件载入token，并对字符进行编码。文件的基本格式为：
    - 每行一个名字
    - 每个名字由若干字符组成
    :return: 二维矩阵[lines_count, max_line_length]，元素就是character对应的编码
    """
    start_token = " "
    with open("names") as f:
        # 按行载入，并在每行之前增加前缀：start_token
        lines = f.read()[:-1].split('\n')
        lines = [start_token + line for line in lines]

        # 统计每行的长度，找到最大值
        max_line_len = max(map(len, lines))
        print("max length =", max_line_len)

        # 得到每行长度的直方图
        plt.title('Sequence length distribution')
        plt.hist(list(map(len, lines)), bins=25)

        # 对每行中的每个字符进行编码，基本过程有两个
        # - 统计所有行出现的所有字符，并唯一化。保存在tokens中。
        # - 给每个字符赋予一个整数值。以字典的形式保存，key是character，value是编码也就是在tokens中的索引，保存在tokens_to_ids中。
        characters = ""
        for line in lines:
            characters += line
        tokens = list(set(characters))
        print("tokens: ", tokens)
        tokens_to_ids = {v: i for i, v in enumerate(tokens)}

        return lines, max_line_len, tokens_to_ids


def to_matrix(lines, encoder, max_line_len=None, pad=0, dtype='int32', batch_first=True):
    """ 将文本内容改造为数值型的矩阵
    :param lines: 列表形式的文本内容，每个元素对应于文本中的行
    :param encoder: 字典，记录字符到整型的映射关系
    :param max_line_len: 行的最大长度，作为矩阵的列数
    :param pad: 每行没有字符的默认编码
    :param dtype: 数据类型
    :param batch_first: 矩阵的形式，是否需要转置
    :return: 文本的二维编码矩阵
    """
    max_line_len = max_line_len or max(map(len, lines))
    lines_ix = np.zeros([len(lines), max_line_len], dtype) + pad

    for i in range(len(lines)):
        line_ix = [encoder[c] for c in lines[i]]
        lines_ix[i, :len(line_ix)] = line_ix

    if not batch_first:  # convert [batch, time] into [time, batch]
        lines_ix = np.transpose(lines_ix)

    return lines_ix
