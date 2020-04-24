import numpy as np
import matplotlib.pyplot as plt

np.random.seed(21)


def generate_bandits(n):
    means = np.random.randn(n)
    return means


class EpsilonGreedyStrategy:

    def __init__(self, eps, true_means):
        self.eps = eps
        self.sigma = 0.1
        self.true_means = true_means
        self.n = len(true_means)
        self.rewards = []
        self.estimates = np.zeros(self.n)
        self.counts = np.zeros(self.n)

    def pull_slot_many_times(self, T):
        for _ in range(T):
            self.pull_slot()

    def pull_slot(self):
        r = np.random.rand()
        if r < self.eps:
            self.explore()
        else:
            self.exploit()

    def exploit(self):
        choice = np.argmax(self.estimates)
        self.play(choice)

    def explore(self):
        choice = np.random.randint(self.n)
        self.play(choice)

    def play(self, choice):
        noise = self.sigma * np.random.randn()
        reward = self.true_means[choice] + noise
        self.rewards += [reward]
        prev_estimate = self.estimates[choice] * self.counts[choice]
        self.estimates[choice] = (prev_estimate + reward) / (self.counts[choice] + 1)
        self.counts[choice] += 1

    def compute_avg_reward(self):
        numerator = np.cumsum(self.rewards)
        denominator = np.cumsum(np.ones_like(self.rewards))
        return numerator / denominator


def run_bandit_experiment(eps_grid, arms, n_trials=1000, T=500, n_actions=10):
    results = np.zeros((4, T))
    for i, eps in enumerate(eps_grid):
        for _ in range(n_trials):
            strat = EpsilonGreedyStrategy(eps=eps, true_means=arms)
            strat.pull_slot_many_times(T)
            avg = strat.compute_avg_reward()
            results[i, :] += avg / n_trials
    return results

def plot_bandit_experiment(results, eps_grid):
    fig, ax = plt.subplots()

    for i, eps in enumerate(eps_grid):
        ax.plot(results[i, :], label='$ \epsilon $: {0:.2f}'.format(eps))
        ax.legend()

    ax.set_title('Comparison of $ \epsilon $-greedy strategies')
    ax.set_xlabel('t')
    ax.set_ylabel('Mean reward')
    plt.show()

def main():
    eps_grid = [0]
    arms = [0.5, 0.4]
    results = run_bandit_experiment(eps_grid, arms)
    plot_bandit_experiment(results, eps_grid)

if __name__ == '__main__':
    print("start")
    main()