import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.nn import functional as F
import numpy as np
from char_rnn import CharRNNCell
from load_data import load_data, to_matrix
from matplotlib import pyplot as plt


def rnn_loop(char_rnn, batch_ix):
    """
    将字符矩阵，按序使用神经网络预测
    """
    batch_size, max_length = batch_ix.size()
    hid_state = char_rnn.initial_state(batch_size)
    logprobs = []

    for x_t in batch_ix.transpose(0, 1):
        hid_state, logp_next = char_rnn(x_t, hid_state)
        logprobs.append(logp_next)

    return torch.stack(logprobs, dim=1)


if __name__ == '__main__':
    # 载入数据
    lines, max_line_len, tokens_to_ids = load_data()
    dataset = to_matrix(lines, tokens_to_ids, max_line_len, pad=tokens_to_ids[' '])
    num_tokens = len(tokens_to_ids)

    # 创建神经网络
    rnn = CharRNNCell(num_tokens)

    # 创建优化器
    opt = torch.optim.Adam(rnn.parameters())
    history = []

    for i in range(1000):
        batch_ix = np.random.choice(len(lines), 32)
        batch_ix = torch.tensor(dataset[batch_ix, :], dtype=torch.int64)
        logp_seq = rnn_loop(rnn, batch_ix)

        # compute loss
        t_logp = logp_seq[:, :-1].contiguous().view(-1, num_tokens)
        t_batch = batch_ix[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(t_logp, t_batch)

        # train with backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append(loss.data.numpy())
        if (i + 1) % 100 == 0:
            plt.clf()
            plt.plot(history, label='loss')
            plt.legend()
            plt.show()
