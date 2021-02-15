import numpy as np
from matplotlib import pyplot as plt
from utility import moving_average

"""
在奖励和状态转移概率等model-free场景下， 这里针对一个名为river swim的简单应用进行演练。

并使用Q-Learning方法作为基础参考，说明其只能获得次优解。更好地基于Posterior Sampling的
解决方案在psrl.py中实现。
"""


class RiverSwimEnv:
    LEFT_REWARD = 5.0 / 1000
    RIGHT_REWARD = 1.0

    def __init__(self, intermediate_states_count=4, max_steps=16):
        self._max_steps = max_steps
        self._current_state = None
        self._steps = None
        self._interm_states = intermediate_states_count
        self.reset()

    def reset(self):
        self._steps = 0
        self._current_state = 1
        return self._current_state, 0.0, False

    @property
    def n_actions(self):
        return 2

    @property
    def n_states(self):
        return 2 + self._interm_states

    def _get_transition_probs(self, action):
        if action == 0:
            if self._current_state == 0:
                return [0, 1.0, 0]
            else:
                return [1.0, 0, 0]

        elif action == 1:
            if self._current_state == 0:
                return [0, .4, .6]
            if self._current_state == self.n_states - 1:
                return [.4, .6, 0]
            else:
                return [.05, .6, .35]
        else:
            raise RuntimeError(
                "Unknown action {}. Max action is {}".format(action, self.n_actions))

    def step(self, action):
        """
        :param action:
        :type action: int
        :return: observation, reward, is_done
        :rtype: (int, float, bool)
        """
        reward = 0.0

        if self._steps >= self._max_steps:
            return self._current_state, reward, True

        transition = np.random.choice(
            range(3), p=self._get_transition_probs(action))
        if transition == 0:
            self._current_state -= 1
        elif transition == 1:
            pass
        else:
            self._current_state += 1

        if self._current_state == 0:
            reward = self.LEFT_REWARD
        elif self._current_state == self.n_states - 1:
            reward = self.RIGHT_REWARD

        self._steps += 1
        return self._current_state, reward, False


class QLearningAgent:
    """ 基于epsilon greedy的q-learning

    主要目的在于说明：探索程度较差，往往得到次优解，特别注意plot_policy的结果，右半部分都没有数据，也就是0，根本没有探索到。
    """

    def __init__(self, n_states, n_actions, lr=0.2, gamma=0.95, epsilon=0.1):
        self._gamma = gamma
        self._epsilon = epsilon
        self._q_matrix = np.zeros((n_states, n_actions))
        self._lr = lr

    def get_action(self, state):
        if np.random.random() < self._epsilon:
            return np.random.randint(0, self._q_matrix.shape[1])
        else:
            return np.argmax(self._q_matrix[state])

    def get_q_matrix(self):
        """ Used for policy visualization
        """

        return self._q_matrix

    def start_episode(self):
        """ Used in PSRL agent
        """
        pass

    def update(self, state, action, reward, next_state):
        """
        Q'(s,a) = r + gamma*max(Q(s',a'))
        Q(s,a) = alpha*Q'(s,a) + (1-alpha)*Q(s,a)
        :param state: 当前状态
        :param action: 当前状态所采取的动作
        :param reward: 当前状态实施动作后获取的奖励
        :param next_state: 当前状态实施动作后进入的下一个状态
        :return: 无需返回值
        """
        qv_next = reward + self._gamma * np.max(self._q_matrix[next_state])
        self._q_matrix[state][action] = self._lr * qv_next + (1 - self._lr) * self._q_matrix[state][action]


def train_mdp_agent(agent, env, n_episodes):
    episode_rewards = []

    for ep in range(n_episodes):
        state, ep_reward, is_done = env.reset()
        agent.start_episode()
        while not is_done:
            action = agent.get_action(state)

            next_state, reward, is_done = env.step(action)
            agent.update(state, action, reward, next_state)

            state = next_state
            ep_reward += reward

        episode_rewards.append(ep_reward)
    return episode_rewards


def plot_policy(agent):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.matshow(agent.get_q_matrix().T)
    #ax.set_yticklabels(['', 'left', 'right'])
    plt.xlabel("State")
    plt.ylabel("Action")
    plt.title("Values of state-action pairs")
    plt.show()


if __name__ == '__main__':
    env = RiverSwimEnv()
    agent = QLearningAgent(env.n_states, env.n_actions)
    rews = train_mdp_agent(agent, env, 1000)
    plt.figure(figsize=(15, 8))

    plt.plot(moving_average(np.array(rews), alpha=.1))
    plt.xlabel("Episode count")
    plt.ylabel("Reward")
    plt.show()

    plot_policy(agent)
