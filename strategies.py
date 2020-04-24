from experiment_base import *
import math


"""This file includes all strategies the experiments will evaluate. 

Normal pull strategies are implemented as subclasses of FreePullBandit.
- random pull
- epsidon greedy
- upper confidence bound
- Thompson sampling

Free pull strategies are implemented as functions accepting FreePullBandit as the argument.
Some uses high-order functions.
- fixed arm
- best arm
- worst arm
- second best arm
- etc.
"""


class RandomPull(FreePullBandit):
    def normal_pull(self):
        # initialization of pulls
        result = self.get_unexplored_arm()
        if result is 0 or result:
            arm = result
        else:
            arm = self.random_pull()
        return arm


class EpsilonGreedy(FreePullBandit):
    epsilon = None

    def __init__(self, device, rounds, k, free_policy=None, epsilon=None):
        super(EpsilonGreedy, self).__init__(device, rounds, k, free_policy)
        if epsilon is None:
            epsilon = EpsilonGreedy.epsilon
            assert epsilon is not None
        self.epsilon = self.original_epsilon = epsilon
        # TODO: it seems temporarily unnecessary to adust epsilon for k
        # if k is None or k < 1:
        #     self.epsilon = self.original_epsilon
        # else:
        #     self.epsilon = (self.original_epsilon * (self.k + 1) - 1) / k
        #     # self.epsilon = (k * self.original_epsilon - 1) / k
        # print(f"adjusted eplision is {self.epsilon}")

    def normal_pull(self):
        # initialization of pulls
        unexplored = self.get_unexplored_arm()
        if unexplored is not None:
            arm = unexplored
        else:
            if np.random.rand() >= self.epsilon:
                scores = [x[0] for x in self.normal_pull_record]
                arm = scores.index(max(scores))
            else:
                arm = self.random_pull()
        return arm


# UCB
class UpperConfidenceBound(FreePullBandit):
    def __init__(self, device, rounds, k, free_policy=None):
        super(UpperConfidenceBound, self).__init__(device, rounds, k, free_policy)

    def normal_pull(self):
        # initialization of pulls
        unexplored = self.get_unexplored_arm()
        if unexplored is not None:
            arm = unexplored
        else:
            scores = [compute_ucb(mean, times, self.T) for (mean, times) in self.normal_pull_record]
            arm = scores.index(max(scores))
        return arm


# Thompson sampling
class ThompsonSampling(FreePullBandit):
    def __init__(self, device, rounds, k, free_policy=None):
        super(ThompsonSampling, self).__init__(device, rounds, k, free_policy)

    def normal_pull(self):
        return np.argmax(np.random.beta(a=self.a, b=self.b))


# free pull strategies
# Funtions named with 'generator' are high-order functions, 
# returning a function of signature (FreePullBandit) => int.
# Arguments called bandit all should be FreePullBandit instances.

def choose_nothing(bandit):
    return


def certain_arm(arm):
    def fixed_policy(bandit):
        return arm
    return fixed_policy


# worst_detection policies: should improve performance
def real_worst_generator(arms):
    arm = arms.index(min(arms))

    def real_worst(bandit):
        return arm
    return real_worst


def worst_mean(bandit):
    scores = [x[0] for x in bandit.normal_pull_record]
    arm = scores.index(min(scores))
    return arm


def least_pulled(bandit):
    scores = [x[1] for x in bandit.normal_pull_record]
    arm = scores.index(min(scores))
    return arm


def ucb_worst(bandit):
    unexplored = bandit.get_unexplored_arm()
    if unexplored is not None:
        arm = unexplored
    else:
        scores = [compute_ucb(mean, times, bandit.T) for (mean, times) in bandit.normal_pull_record]
        arm = scores.index(max(scores))
    return arm


def ts_worst(bandit):
    return np.argmin(np.random.beta(a=bandit.a, b=bandit.b))


# Worst under successive elimination free policy
def worst_ucb_with_successive_elimination(bandit):
    scores = [compute_se_ucb(bandit, i) for i in range(bandit.arm_num)]
    best = scores.index(max(scores))
    low_bound = bandit.normal_pull_record[best][0]*2-scores[best]
    min_i, min_s = best, scores[best]
    for i, s in enumerate(scores):
        if s <= low_bound:
            bandit.rejected[i] = 1
        else:
            if s < min_s:
                min_i, min_s = i, s
    return min_i


def worst_mean_with_successive_elimination(bandit):
    scores = [compute_se_ucb(bandit, i) for i in range(bandit.arm_num)]
    best = scores.index(max(scores))
    low_bound = bandit.normal_pull_record[best][0]*2-scores[best]
    min_i, min_s = best, bandit.normal_pull_record[best][0]
    for i, s in enumerate(scores):
        if s <= low_bound:
            bandit.rejected[i] = 1
        else:
            if bandit.normal_pull_record[i][0] < min_s:
                min_i, min_s = i, bandit.normal_pull_record[i][0]
    return min_i


def least_pulled_with_successive_elimination(bandit):
    scores = [compute_se_ucb(bandit, i) for i in range(bandit.arm_num)]
    best = scores.index(max(scores))
    low_bound = bandit.normal_pull_record[best][0]*2-scores[best]
    min_i, min_s = best, bandit.normal_pull_record[best][1]
    for i, s in enumerate(scores):
        if s <= low_bound:
            bandit.rejected[i] = 1
        else:
            if bandit.normal_pull_record[i][1] < min_s:
                min_i, min_s = i, bandit.normal_pull_record[i][1]
    return min_i


# best_detection policies
def real_best_generator(arms):
    arm = arms.index(max(arms))

    def real_best(bandit):
        return arm
    return real_best


def best_mean(bandit):
    scores = [x[0] for x in bandit.normal_pull_record]
    arm = scores.index(max(scores))
    return arm


def most_pulled(bandit):
    scores = [x[1] for x in bandit.normal_pull_record]
    arm = scores.index(max(scores))
    return arm


def ucb_best(bandit):
    unexplored = bandit.get_unexplored_arm()
    if unexplored is not None:
        arm = unexplored
    else:
        scores = [compute_ucb(mean, times, bandit.T) for (mean, times) in bandit.normal_pull_record]
        arm = scores.index(max(scores))
    return arm


def ts_best(bandit):
    return np.argmax(np.random.beta(a=bandit.a, b=bandit.b))


# second_best detection policies: might harm performance
def real_second_best_generator(arms):
    arm = get_second_max(arms)

    def real_second_best(bandit):
        return arm
    return real_second_best


def second_best_mean(bandit):
    scores = [x[0] for x in bandit.normal_pull_record]
    return get_second_max(scores)


def second_most_pulled(bandit):
    scores = [x[1] for x in bandit.normal_pull_record]
    return get_second_max(scores)


def ucb_second_best(bandit):
    unexplored = bandit.get_unexplored_arm()
    if unexplored is not None:
        arm = unexplored
    else:
        scores = [compute_ucb(mean, times, bandit.T) for (mean, times) in bandit.normal_pull_record]
        arm = get_second_max(scores)
    return arm


def ts_second_best(bandit):
    return np.argpartition(np.random.beta(a=bandit.a, b=bandit.b), -2)[-2]


# ranked best policies
def real_kth_best_generator(arms, k):
    arm = get_kth_max(arms, k)

    def real_kth_best(bandit):
        return arm
    return real_kth_best


# same policy
def epsilongreedy_policy_generator(epsilon):
    def epsilongreedy(bandit):
        if np.random.rand() >= epsilon:
            scores = [x[0] for x in bandit.normal_pull_record]
            arm = scores.index(max(scores))
        else:
            arm = bandit.random_pull()
        return arm
    return epsilongreedy


def policies_generator(arms):
    default_policy = {
        "pure_explore": None
    }

    worst_policies = {
        "real_worst": real_worst_generator(arms),
        "worst_mean": worst_mean,
        "least_pulled": least_pulled,
        "ucb_worst": ucb_worst,
        "ts_worst": ts_worst
    }

    best_policies = {
        "real_best": real_best_generator(arms),
        "best_mean": best_mean,
        "most_pulled": most_pulled,
        "ucb_best": ucb_best,
        "ts_best": ts_best
    }

    second_best_policies = {
        "real_second_best": real_second_best_generator(arms),
        "second_best_mean": second_best_mean,
        "second_most_pulled": second_most_pulled,
        "ucb_second_best": ucb_second_best,
        "ts_second_best": ts_second_best
    }

    all_policies = {**default_policy, **worst_policies, **best_policies, **second_best_policies}

    seperated_policies_dict = {
        "worst_policies": worst_policies,
        "best_policies": best_policies,
        "second_best_policies": second_best_policies
    }
    return seperated_policies_dict
