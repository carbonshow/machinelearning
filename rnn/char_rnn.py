import torch, torch.nn as nn
import torch.nn.functional as F


class CharRNNCell(nn.Module):
    """ 字符处理的神经网络定义

    - 输入是字符编码。通常是one hot编码矢量，最大长度很长但只有少数内容有效
    - 添加embedding layer。实际上是生成一个查找表，用于降维。可以参考：https://www.zhihu.com/question/45027109
    - 添加输出层，用来预测下一个音素（phoneme）出现的概率
    """

    def __init__(self, num_tokens, embedding_size=16, rnn_num_units=64):
        super(self.__class__, self).__init__()
        self.num_units = rnn_num_units

        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.rnn_update = nn.Linear(embedding_size + rnn_num_units, rnn_num_units)
        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, x, h_prev):
        """
        This method computes h_next(x, h_prev) and log P(x_next | h_next)
        We'll call it repeatedly to produce the whole sequence.

        :param x: batch of character ids, int64[batch_size]
        :param h_prev: previous rnn hidden states, float32 matrix [batch, rnn_num_units]
        """
        # get vector embedding of x
        x_emb = self.embedding(x)

        # compute next hidden state using self.rnn_update
        # hint: use torch.cat(..., dim=...) for concatenation
        h_next = self.rnn_update(torch.cat((x_emb, h_prev), 1))

        h_next = torch.tanh(h_next)

        assert h_next.size() == h_prev.size()

        # compute logits for next character probs
        logits = self.rnn_to_logits(h_next)

        return h_next, F.log_softmax(logits, -1)

    def initial_state(self, batch_size):
        """ return rnn state before it processes first input (aka h0) """
        return torch.zeros(batch_size, self.num_units)
