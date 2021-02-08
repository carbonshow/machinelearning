import numpy as np
from bernoulli_bandit import BernoulliBandit


class DriftingBandit(BernoulliBandit):
    """ 每条臂获得奖励的概率随时间变化
    这是经典的non stationary bandit问题，当然采用了一种比较固定的方式对每个摇臂的奖励概率进行调整。

    解决这种问题一般有两种思路：
    - 对奖励进行衰减，减少早起行为和奖励对当前概率评估的影响。因为概率是变化的，所以尽可能选择靠近当前的采样，来作为最新的概率估计。
    - 感知环境的变化，在探测到变化后充值算法或者显式要求进行探索以重新获得对环境的评估
    """

    def __init__(self, n_actions=5, gamma=0.01):
        """
        Idea from https://github.com/iosband/ts_tutorial
        """
        super().__init__(n_actions)

        self._gamma = gamma

        self._successes = None
        self._failures = None
        self._steps = 0

        self.reset()

    def reset(self):
        self._successes = np.zeros(self.action_count) + 1.0
        self._failures = np.zeros(self.action_count) + 1.0
        self._steps = 0

    def step(self):
        action = np.random.randint(self.action_count)
        reward = self.pull(action)
        self._step(action, reward)

    def _step(self, action, reward):
        self._successes = self._successes * (1 - self._gamma) + self._gamma
        self._failures = self._failures * (1 - self._gamma) + self._gamma
        self._steps += 1

        self._successes[action] += reward
        self._failures[action] += 1.0 - reward

        self._probs = np.random.beta(self._successes, self._failures)
