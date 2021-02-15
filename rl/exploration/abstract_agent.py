from abc import ABCMeta, abstractmethod

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# 定义Agent基础类型，用于根据环境信息更新自己的策略，并选择动作
class AbstractAgent(metaclass=ABCMeta):
    """ Agent基础抽象类

    定义Agent基础类型，用于根据环境信息更新自己的策略，并选择动作。
    提供的基础功能：控制整个摇臂，记录奖励的流程
    抽象接口：选择动作的策略，决定了探索与利用的平衡方法。
    """

    def init_actions(self, n_actions):
        """ 为每个动作初始化统计数据

        self._successes 数组self._successes[k]即alpha(k)，表示当前状态下，动作k的累计成功次数
        self._failures 数组self._failures[k]即beta(k)，表示当前状态下，动作k的累计失败次数

        :param n_actions: 动作空间的大小
        """
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._actions_cnt = n_actions
        self._total_pulls = 0

    @abstractmethod
    def get_action(self):
        """
        Get current best action
        :rtype: int
        """
        pass

    def update(self, action, reward):
        """
        Observe reward from action and update agent's internal parameters
        :type action: int
        :type reward: int
        """
        self._total_pulls += 1
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1

    @property
    def name(self):
        return self.__class__.__name__


