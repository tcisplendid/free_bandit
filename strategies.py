from experiment_base import *
import math


# pragma: normal pull strategies

# epsilon greedy
class EpsilonGreedy(FreePullBandit):
    epsilon = None

    def __init__(self, device, rounds, k, free_policy=None, epsilon=None):
        super(EpsilonGreedy, self).__init__(device, rounds, k, free_policy)
        if epsilon is None:
            epsilon = EpsilonGreedy.epsilon
            assert epsilon is not None
        self.epsilon = self.original_epsilon = epsilon
        # if k is None or k < 1:
        #     self.epsilon = self.original_epsilon
        # else:
        #     self.epsilon = (self.original_epsilon * (self.k + 1) - 1) / k
        #     # self.epsilon = (k * self.original_epsilon - 1) / k
        # print(f"adjusted eplision is {self.epsilon}")

    def normal_pull(self):
        # initialization of pulls
        result = self.has_arm_unexplored()
        if result is 0 or result:
            arm = result
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
        def compute_ucb(arm):
            mean, times = self.normal_pull_record[arm]
            # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
            confidence = math.sqrt(2 * math.log(self.T) / float(times))
            return mean + confidence

        # initialization of pulls
        result = self.has_arm_unexplored()
        if result is 0 or result:
            arm = result
        else:
            scores = [compute_ucb(i) for i in range(self.arm_num)]
            arm = scores.index(max(scores))
        return arm


# Thompson sampling
class ThompsonSampling(FreePullBandit):
    def __init__(self, device, rounds, k, free_policy=None):
        super(ThompsonSampling, self).__init__(device, rounds, k, free_policy)

    def normal_pull(self):
        return np.argmax(np.random.beta(a=self.a, b=self.b))


# pragma: free pull strategies

def certain_arm(arm):
    def fixed_policy(self):
        return arm
    return fixed_policy


def get_second_max(arr):
    copy = arr[:]
    copy.remove(max(arr))
    return arr.index(max(copy))


# worst_detection policies: should improve performance

def real_worst_generator(arms):
    arm = arms.index(min(arms))

    def real_worst(self):
        return arm
    return real_worst


def worst_mean(self):
    scores = [x[0] for x in self.normal_pull_record]
    arm = scores.index(min(scores))
    return arm


def least_pulled(self):
    scores = [x[1] for x in self.normal_pull_record]
    arm = scores.index(min(scores))
    return arm


def ucb_worst(self):
    def compute_ucb(arm):
        mean, times = self.normal_pull_record[arm]
        # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
        confidence = math.sqrt(2*math.log(self.T)/float(times))
        return mean + confidence

    result = self.has_arm_unexplored()
    if result is 0 or result:
        arm = result
    else:
        scores = [compute_ucb(i) for i in range(self.arm_num)]
        arm = scores.index(min(scores))
    return arm


def ts_worst(self):
    return np.argmin(np.random.beta(a=self.a, b=self.b))


# best_detection policies:
def real_best_generator(arms):
    arm = arms.index(max(arms))

    def real_best(self):
        return arm
    return real_best


def best_mean(self):
    scores = [x[0] for x in self.normal_pull_record]
    arm = scores.index(max(scores))
    return arm


def most_pulled(self):
    scores = [x[1] for x in self.normal_pull_record]
    arm = scores.index(max(scores))
    return arm


def ucb_best(self):
    def compute_ucb(arm):
        mean, times = self.normal_pull_record[arm]
        # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
        confidence = math.sqrt(2*math.log(self.T)/float(times))
        return mean + confidence

    result = self.has_arm_unexplored()
    if result is 0 or result:
        arm = result
    else:
        scores = [compute_ucb(i) for i in range(self.arm_num)]
        arm = scores.index(max(scores))
    return arm


def ts_best(self):
    return np.argmax(np.random.beta(a=self.a, b=self.b))


# second_best detection policies: might harm performance

def real_second_best_generator(arms):
    arm = get_second_max(arms)

    def real_second_best(self):
        return arm
    return real_second_best


def second_best_mean(self):
    scores = [x[0] for x in self.normal_pull_record]
    return get_second_max(scores)


def second_most_pulled(self):
    scores = [x[1] for x in self.normal_pull_record]
    return get_second_max(scores)


def ucb_second_best(self):
    def compute_ucb(arm):
        mean, times = self.normal_pull_record[arm]
        # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
        confidence = math.sqrt(2*math.log(self.T)/float(times))
        return mean + confidence

    result = self.has_arm_unexplored()
    if result is 0 or result:
        arm = result
    else:
        scores = [compute_ucb(i) for i in range(self.arm_num)]
        arm = get_second_max(scores)
    return arm


def ts_second_best(self):
    return np.argpartition(np.random.beta(a=self.a, b=self.b), -2)[-2]


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