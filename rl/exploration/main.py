from agents import EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent, DiscountThompsonSamplingAgent
from bernoulli_bandit import BernoulliBandit
from drifting_bandit import DriftingBandit
from regret import get_regret, plot_regret


def compare_agents_fix_bernoulli(agents):
    regret = get_regret(BernoulliBandit(), agents, n_steps=10000, n_trials=50)
    plot_regret(agents, regret)


def compare_agents_drifting_bernoulli(agents):
    regret = get_regret(DriftingBandit(), agents, n_steps=10000, n_trials=50)
    plot_regret(agents, regret)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Use a breakpoint in the code line below to debug your script.
    target_agents = [
        EpsilonGreedyAgent(0.01),
        #EpsilonGreedyAgent(0.2, 0.7),
        UCBAgent(),
        ThompsonSamplingAgent(),
        DiscountThompsonSamplingAgent(0.995),
    ]

    #compare_agents_fix_bernoulli(target_agents)
    compare_agents_drifting_bernoulli(target_agents)

