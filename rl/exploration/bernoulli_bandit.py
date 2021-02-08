import numpy as np


class BernoulliBandit:
    """ 伯努摇臂机

    每个动作（摇臂）产生奖励符合伯努利分布：要么产生奖励1，要么产生奖励0。每个摇臂产生奖励的概率各不相同。

    假设第i个动作（摇臂）产生奖励的概率为p(i)，意味着有p(i)的概率产生奖励1，1-p(i)的概率生成奖励0.
    """

    def __init__(self, n_actions=5):
        self._probs = np.random.random(n_actions)

    @property
    def action_count(self):
        return len(self._probs)

    def pull(self, action):
        if np.any(np.random.random() > self._probs[action]):
            return 0.0
        return 1.0

    def optimal_reward(self):
        """ Used for regret calculation
        """
        return np.max(self._probs)

    def step(self):
        """ Used in nonstationary version
        """
        pass

    def reset(self):
        """ Used in nonstationary version
        """
