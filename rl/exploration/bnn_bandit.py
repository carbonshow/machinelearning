import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from utility import moving_average

"""
基本思路是：
1. 搭建bayesian neural network，根据状态，预测动作。
2. 状态是多维矢量；动作会有固定数量的候选，每次只能选择其中的一个，因此属于典型的分类问题
3. 样本集已经给出了明确的状态空间和最优动作之间的映射关系
4. 使用cross entropy和KL-divergence的组合作为代价函数，对神经网络剧进行梯度下降的训练
"""


class BNNThompsonAgent:
    """ a bandit with bayesian neural net
    基于pytorch搭建bayesian神经网络，使用的库是torchbnn，github地址为：https://github.com/Harry24k/bayesian-neural-network-pytorch

    这里使用的是Thompson Sampling，也就是利用BNN根据输入状态做一次采样；也可以使用多次采样求期望或者分位数的方式。
    """

    def __init__(self, state_features_size, action_features_size):
        self.n_states = state_features_size
        self.n_actions = action_features_size
        self._create_nn_model()
        self.epsilon = 0.25

    def _create_nn_model(self):
        print("in_features: {}, out_features: {}".format(self.n_states, self.n_actions))
        self.model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.n_states, out_features=100),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=self.n_actions),
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.kl_weight = 0.1

    def get_action(self, pred_actions):
        """ 基于epsilon-greedy选择动作
        - 1-epsilon的概率，利用
        - epsilon的概率, 探索
        核心在于：对（状态、动作）对应的奖励期望进行比较，视期望最高的动作为当前状态下的最优动作

        其实也完全可以使用：thompson，UCB等方法。核心区别在于——是否需要多次采样：
        - 只有一次，那么认为是基于BNN的Thompson采样，传统是基于Beta分布
        - 多次采样，又可以分为两种：
          - 求期望，作为该动作的最终reward。
          - 使用指定分位数对应的值。其实就是UCB

        :param pred_actions: 针对输入状态，由神经网络输出的各个actions分类中的数值
        :returns: 矢量，记录states中各个状态对应的，选择的动作
        """

        # 针对每个记录，选择值最高的action作为最优action：argmax(axis=-1)
        best_actions = pred_actions.argmax(axis=-1)

        # 每个状态独立使用epsilon-greedy策略
        return [np.random.randint(0, n_actions) if np.random.random() < self.epsilon else best_actions[i] for i in
                range(len(best_actions))]

    def train(self, policy_actions, optimal_actions):
        """
        针对指定的状态集合，训练预测相关动作所获得的奖励

        :param rewards: 实际获得的奖励集合 [batch_size, 1]
        """
        loss_sum = kl_sum = 0
        # 根据状态预测动作，计算对应的奖励，然后将该奖励和实际奖励进行对比
        ce = self.ce_loss(policy_actions, optimal_actions)
        kl = self.kl_loss(self.model)
        cost = ce + self.kl_weight * kl

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        loss_sum += cost
        kl_sum += kl

        return ce, kl

    def predict(self, input_states):
        return self.model(torch.from_numpy(input_states).float())

    @property
    def name(self):
        return self.__class__.__name__


def get_new_samples(states, act_rewards, batch_size=10):
    """ 返回action、reward采样数据
    从action_rewards中随机采样指定数量的记录，类似于Experience Replay的思想

    :returns: tuple(states, actions), states: [batch_size, n_states], actions: [batch_size, n_actions]
    """
    batch_ix = np.random.randint(0, len(states), batch_size)
    return states[batch_ix], act_rewards[batch_ix]


def train_contextual_agent(agent, record_states, record_action_rewards, batch_size=10, n_iters=100):
    """ 基于神经网络训练agent

    :param agent: 神经网络Agent
    :param batch_size: 每轮迭代训练采样的批量数据集合的大小
    :param n_iters: 迭代次数
    :return: 奖励的移动平均结果
    """
    rewards_history = []

    for i in range(n_iters):
        # 采样生成batch_size对应的样本集合，返回获得的状态，对应的动作、奖励。相互之间均是一一对应的关系
        # 这些都是样本数据，也可以认为是最优策略，即给定状态，返回optimal action and the corresponding reward
        sample_states, sample_action_rewards = get_new_samples(
            record_states, record_action_rewards, batch_size)

        # 根据状态进行预测，返回分类的预估结果
        pred_actions = agent.predict(sample_states)

        # 根据epsilon-greedy策略获得当前计算出来的动作，以及实际获得的奖励
        policy_actions = agent.get_action(pred_actions)
        policy_rewards = sample_action_rewards[
            np.arange(batch_size), policy_actions
        ]

        # 根据states获得最优actions
        optimal_actions = torch.tensor([np.argmax(sample_action_rewards[row]) for row in range(batch_size)])

        # 训练模型，梯度下降并计算预测结果和样本最优结果之间的差异，也就是更新损失：均方根和KL散度
        mse, kl = agent.train(pred_actions, optimal_actions)

        # 每个迭代的loss记录下来，便于作图并观察变化趋势。
        rewards_history.append(policy_rewards.mean())

        if i % 10 == 0:
            # clear_output(True)
            plt.clf()
            print("iteration #%i\tmean reward=%.3f\tmse=%.3f\tkl=%.3f" %
                  (i, np.mean(rewards_history[-10:]), mse, kl))
            plt.plot(rewards_history)
            plt.plot(moving_average(np.array(rewards_history), alpha=0.1))
            plt.title("Reward per episode")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.show()


if __name__ == '__main__':
    # 从文件中加载状态、动作和奖励数据
    all_states = np.load("all_states.npy")  # 所有的状态数据，来自all_states.npy文件，返回的数据是二维数组[batch_size, n_states]
    action_rewards = np.load("action_rewards.npy")  # 素有的动作奖励数据，来自action_rewards.npy文件，返回的数据是二维数组[batch_size, n_actions]
    state_size = all_states.shape[1]  # 状态空间大小，载入 all_states文件后设置
    n_actions = action_rewards.shape[1]  # 动作空间大小，载入action_rewards文件后设置
    n_iterations = 200  # 训练的总迭代次数

    print("State size: %i, actions: %i" % (state_size, n_actions))
    print(action_rewards)

    # 创建神经网络
    bnn_agent = BNNThompsonAgent(state_size, n_actions)

    # 训练
    train_contextual_agent(bnn_agent, all_states, action_rewards, 100, n_iterations)
