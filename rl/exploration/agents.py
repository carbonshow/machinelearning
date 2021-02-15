import sys
import numpy as np
import math
from abstract_agent import AbstractAgent

"""
定义各种Agents，用于明确不同 "探索与利用" 策略的差异，并相互比较，目前主要有：Random，Epsilon-Greedy，Thompson Sampling，UCB
"""


class RandomAgent(AbstractAgent):
    """ 随机探索与利用

    最简单的Agent，依靠全随机来选择动作，不会主动利用当前已知的最优策略，也不会主动探索未知状态。
    """

    def get_action(self):
        return np.random.randint(0, self._actions_cnt)


class EpsilonGreedyAgent(AbstractAgent):
    """ Epsilon-Greedy Agent

    以概率epsilon探索，以概率1-epsilon利用（选择最优动作）。不难发现对探索其实是随机的，缺少对有效探索的倾向。什么是有效探索呢？
    - 那些我们不是很了解的选择。意味着不确定性高一些
    - 那些潜在可能更优的选择
    """

    def __init__(self, epsilon=0.01, decay=1.0):
        self._init_epsilon = epsilon
        self._epsilon = epsilon
        self._decay = decay

    def get_action(self):
        prob = np.random.random()
        actions_weights = np.zeros(self._actions_cnt)

        if prob < self._epsilon or self._total_pulls < 1:
            self._update_epsilon()
            return np.random.randint(0, self._actions_cnt)
        else:
            for action in range(self._actions_cnt):
                action_apply_cnt = self._successes[action] + self._failures[action]

                # 用来衡量探索和利用的的解决取对数小于0，导致平方根无法计算。
                if action_apply_cnt > 0:
                    weight = self._successes[action] / action_apply_cnt
                else:
                    weight = np.random.random(1)[0]
                actions_weights[action] = weight

            self._update_epsilon()
            return np.argmax(actions_weights)

    @property
    def name(self):
        return self.__class__.__name__ + "(epsilon={},decay={})".format(self._init_epsilon, self._decay)

    def _update_epsilon(self):
        self._epsilon *= self._decay


class UCBAgent(AbstractAgent):
    """ UCB(Upper Confidence Bound)

    基本思路是选择：优先选择期望大，方差小的动作。使用UCB1算法：

    - 更新方法 Q'(s,a) = Q(s,a) + alpha * sqrt(2logN(s)/n(s,a))
    - 动作选择方法: 选择让Q'最大的动作a

    for t = 1, 2, ... do
        for k = 1, 2, ... do
            w(k) = a(k)/(a(k)+b(k)) +sqrt(2log(t)/(a(k)+b(k)))
        end for
        x(t) = argmax_k w(k)
        apply x(t) and observe r(t)
        (a(x(t)), b(x(t))) = (a(x(t)), b(x(t))) + (r(t), (1-r(t)))
    end for

    t 表示实施的动作数量，状态变更次数
    k 表示在对应状态下，实施的动作编号
    w(k) 是ucb-1算法的核心，用来记录在当前状态下，实施各个动作k所对应的权重
    x(t) 当前所应实施的动作，也就是w最高的动作
    a,b  是对奖励的更新方法，是伯努利分布下累计奖励的一种表示，针对成功奖励为1，失败奖励为0的情况，a，b就是成功与失败的次数
    """

    def get_action(self):
        """
        计算当前状态下，所有的action对应的权重，然后选择权重最大的动作，作为当前需要选择的动作
        """
        actions_weights = np.zeros(self._actions_cnt)

        for action in range(self._actions_cnt):
            action_apply_cnt = self._successes[action] + self._failures[action]

            # 用来衡量探索和利用的的解决取对数小于0，导致平方根无法计算。
            if action_apply_cnt > 0:
                bias = 0.0 if self._total_pulls <= 1 else math.sqrt(2 * math.log(self._total_pulls) / action_apply_cnt)
                weight = self._successes[action] / action_apply_cnt + bias
            else:
                weight = np.random.random(1)[0]
            actions_weights[action] = weight

        return np.argmax(actions_weights)


class ThompsonSamplingAgent(AbstractAgent):
    """ Thompson Sampling
    最基本的方法就是：在当前状态下，统计不同动作成功获得奖励的比例。选择最高的那个动作
    """

    def get_action(self):
        actions_weights = np.zeros(self._actions_cnt)

        for action in range(self._actions_cnt):
            if self._successes[action] > 0 and self._failures[action] > 0:
                weight = np.random.beta(self._successes[action], self._failures[action])
            else:
                weight = np.random.random(1)[0]
            actions_weights[action] = weight

        return np.argmax(actions_weights)


class DiscountThompsonSamplingAgent(ThompsonSamplingAgent):
    """ Discounting Thompson Sampling
    选择摇臂的方式和基础的thompson sampling一致：基于Beta分布，但是在更新摇臂分布参数的方法上需要考虑到折扣因素
    """

    def __init__(self, gamma=0.9):
        super().__init__()
        self._gamma = gamma

    def init_actions(self, n_actions):
        super().init_actions(n_actions)
        self._successes.fill(1.0)
        self._failures.fill(1.0)

    def update(self, action, reward):
        self._total_pulls += 1
        # 首先对所有摇臂的记录进行衰减
        self._successes = self._successes*self._gamma
        self._failures = self._failures*self._gamma

        # 对于被选中的摇臂，要考虑折扣因素，并更新
        self._successes[action] += reward
        self._failures[action] += 1 - reward

    @property
    def name(self):
        return self.__class__.__name__ + "(gamma={})".format(self._gamma)
