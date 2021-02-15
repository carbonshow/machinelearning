import numpy as np
from matplotlib import pyplot as plt
from utility import moving_average
from river_swim import RiverSwimEnv
from river_swim import train_mdp_agent, plot_policy

"""
和Q-Learning比较，说明如何更好地解决river swim问题。基于Posterior Thompson Sampling实现。
应该会有更好地探索方式，更利于获得最优解。

原论文见：(More) Efficient Reinforcement Learning via Posterior Sampling
"""


def sample_normal_gamma(mu, lmbd, alpha, beta):
    """ https://en.wikipedia.org/wiki/Normal-gamma_distribution

    基本思想是，对于未知期望和方差的正态分布，基于贝叶斯使用NormalGamma作为其共轭后验分布：
    (X,T) ~ NormalGamma(mu,lambda,alpha,beta)
    其中T符合alpha和beta决定的Gamma分布
    X符合N(mu, 1/(lambda*t))正态分布
    """
    tau = np.random.gamma(alpha, beta)
    mu = np.random.normal(mu, 1.0 / np.sqrt(lmbd * tau))
    return mu, tau


class PsrlAgent:
    """
    Posterior Sampling for Reinforcement Learning

    算法伪代码如下：
    for episode k=1,2,... do
      sample M(k) ~ f(.|H(k))
      compute policy mu(k) for M(k)
      for t = 1,2,... do
        take action a(t) from mu(k)
        observe reward r(t) and s(t+1), update H(k)
      end for
    end for

    主要包含对两个目标分布的估计：
    - 状态转移概率分布。假设符合Dirichlet分布，常用作贝叶斯的先验概率，是多变量普遍化的Beta分布。
    - 奖励分布。假设符合NormalGamma分布。


    """

    def __init__(self, n_states, n_actions, gamma=0.95, horizon=10):
        self._n_states = n_states
        self._n_actions = n_actions
        self._gamma = gamma
        self._horizon = horizon

        # params for transition sampling - Dirichlet distribution
        self._transition_counts = np.zeros(
            (n_states, n_states, n_actions)) + 1.0

        # params for reward sampling - Normal-gamma distribution
        self._mu_matrix = np.zeros((n_states, n_actions)) + 1.0
        self._state_action_counts = np.zeros(
            (n_states, n_actions)) + 1.0  # lambda

        self._alpha_matrix = np.zeros((n_states, n_actions)) + 1.0
        self._beta_matrix = np.zeros((n_states, n_actions)) + 1.0

    def _value_iteration(self, transitions, rewards):
        """
        计算值函数，根据论文的定义：计算值函数V(s)，假设采取动作a，可能生成新的状态s'，那么将(s,a,s')的奖励期望计算出来。
        然后选择奖励期望最大的动作对应的奖励，即为V(s)
        :param transitions: 状态动作概率矩阵，通过狄利克雷分布生成
        :param rewards: 状态下采取某个动作所获得的奖励期望
        :return: 状态值函数
        """
        # 基于值迭代的方式，将states_values对所有的状态均初始化为0，然后迭代求解
        state_values = np.zeros(self._n_states)

        def get_new_state_value(state):
            def get_action_value(state_values, state, action):
                """ Computes Q(s,a) as in formula above """
                # perform action 'a' under state 's', the avaiable next states with the probability
                q = 0.0
                for next_state in range(self._n_states):
                    q += transitions[state][next_state][action] * (
                                rewards[state][action] + self._gamma * state_values[next_state])
                return q

            qs = [get_action_value(state_values, state, action) for action in range(self._n_actions)]
            return max(qs)

        for i in range(100):
            new_state_values = np.array([get_new_state_value(s) for s in range(len(state_values))])

            # Compute difference
            diff = max(abs(new_state_values[s] - state_values[s]) for s in range(self._n_states))
            state_values = new_state_values
            if diff < 0.05:
                break
        return state_values

    def start_episode(self):
        # 采样新的mdp过程，沿着next state轴，对current state和action进行狄利克雷采样
        # (n_states, n_states, n_actions)
        self._sampled_transitions = np.apply_along_axis(
            np.random.dirichlet, 1, self._transition_counts)

        # sampled_reward_mus: (n_states, n_actions)
        # sampled_reward_stds: (n_states, n_actions)
        sampled_reward_mus, sampled_reward_stds = sample_normal_gamma(
            self._mu_matrix,
            self._state_action_counts,
            self._alpha_matrix,
            self._beta_matrix
        )

        # (n_states, n_actions)
        self._sampled_rewards = sampled_reward_mus
        self._current_value_function = self._value_iteration(
            self._sampled_transitions, self._sampled_rewards)

    def get_action(self, state):
        """
        选择Q(s,a)最大的动作即可
        :param state: 当前状态
        :return: 动作值函数最大的动作
        """
        return np.argmax(self._sampled_rewards[state] +
                         self._current_value_function.dot(self._sampled_transitions[state]))

    def update(self, state, action, reward, next_state):
        """
        更新共轭先验参数: https://en.wikipedia.org/wiki/Conjugate_prior
        主要有：
        - 狄利克雷更新(离散)
        - normal-gamma更新(连续)

        :param state: 当前状态
        :param action: 实施的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        """
        # 狄利克雷先验参数更新
        self._transition_counts[state][next_state][action] += 1

        # normal-gamma先验参数更新，主要包含四个参数：
        # mu: self._mu_matrix,
        # v: self._state_action_counts,
        # alpha: self._alpha_matrix,
        # beta: self._beta_matrix

        # state，action已经采样过的次数
        sample_count = self._state_action_counts[state][action]
        # state，action累计已经获得的奖励之和
        sum_reward = self._mu_matrix[state][action] * (sample_count + 1) - 1
        # state, action已经获得的奖励期望
        mean_reward = sum_reward / sample_count
        new_mean_reward = (sum_reward + reward) / (sample_count + 1)
        # 更新mu
        self._mu_matrix[state][action] = (1 + sum_reward + reward) / (sample_count + 1)

        # 更新v
        self._state_action_counts[state][action] += 1

        # 更新alpha
        self._alpha_matrix[state][action] += 0.5

        # 更新beta
        # state, action已经获得的奖励的方差
        var_reward = ((self._beta_matrix[state][action] - 1) * 2 - sample_count / (sample_count + 1) * (
                mean_reward - 1) ** 2) / sample_count
        new_var_reward = sample_count / ((sample_count + 1) ** 2) * ((reward - mean_reward) ** 2) + sample_count / (
                sample_count + 1) * var_reward
        self._beta_matrix[state][action] = 1 + 0.5 * (
                (sample_count + 1) * new_var_reward + (sample_count + 1) / (sample_count + 2) * (
                (new_mean_reward - 1) ** 2))

    def get_q_matrix(self):
        return self._sampled_rewards + self._current_value_function.dot(self._sampled_transitions)


if __name__ == '__main__':
    horizon = 20
    env = RiverSwimEnv(max_steps=horizon)
    agent = PsrlAgent(env.n_states, env.n_actions, horizon=horizon)
    rews = train_mdp_agent(agent, env, 1000)

    plt.figure(figsize=(15, 8))
    plt.plot(moving_average(np.array(rews), alpha=0.1))

    plt.xlabel("Episode count")
    plt.ylabel("Reward")
    plt.show()

    plot_policy(agent)
