import numpy as np
from simulator import *
import math
import bisect
import random
import matplotlib.pyplot as plt



"""
This file implements several naive algorithms for Weak/Strong gang of bandits problem:

What can be improved includes (1)flexible allocation of pulls, (2) using vars information.

1. CombinatorialWeakPullAlgo:  treating the gang of bandits as a combinatorial bandit each arm of which is a combination of arms of original bandits
2. PureStrongPullAlgo:  only uses the strong pull. Treat it as multiple simple bandit problem. Pulls are allocated to each device according to its number of arms.
3. StrongModelWeakPullAlgo:  uses the weak pull. Similar as the last algo but utilizing the fact that the reward function is a simple sum().
4. StrongModelWeakStrongALgo: combines 2 and 3.

"""


class CombinatorialWeakPullAlgo(WeakStrongBanditAlgo):
    """
    A UCB algo for a gang of bandits with any reward function.
    """

    def initialize(self):
        super(CombinatorialWeakPullAlgo, self).initialize()
        self.algo_name = "CombinatorialWeakPullAlgo"
        self.weak_arm_num = np.prod(self.arm_nums)
        multipliers = []
        for i in range(self.device_num):
            if i < self.device_num-1:
                multipliers.append(np.prod(self.arm_nums[i+1:]))
            else:
                multipliers.append(1)
        self.__arm_index_multiplier__ = multipliers

    # def get_index_of_weak_arm(self, arms_tuple):

    def next_pull(self):
        self.T += 1
        def compute_UCB(weak_arm):
            mean, times = self.weak_pull_record[weak_arm]
            confidence = math.sqrt(2*math.log(self.T)/float(times))
            return mean + confidence
        if self.T <= self.weak_arm_num:
            arms = []
            chosen_arm = 0
            index = self.T - 1
            for i in range(self.device_num):
                multiplier = self.__arm_index_multiplier__[i]
                chosen_arm = index // multiplier
                index = index % multiplier
                arms.append(chosen_arm)
            return "weak", tuple(arms)
        else:
            arms = list(self.weak_pull_record.keys())
            max_arm = max(arms, key=compute_UCB)
            return "weak", max_arm

    def update_weak_pull_record(self, arms, reward):
        records = self.weak_pull_record
        if arms not in records.keys():
            records[arms] = (0, 0)
        mean, times = records[arms]
        mean = (mean * times + reward) / float(times+1)
        records[arms] = (mean, times+1)


class PureStrongPullAlgo(WeakStrongBanditAlgo):
    """
    Only uses strong pulls. The pulls are uniformly allocated to each device.
    """
    def initialize(self):
        super(PureStrongPullAlgo, self).initialize()
        self.algo_name = "PureStrongPullAlgo"
        self.strong_arm_nums = sum(self.arm_nums)
        self.strong_B = 0
        self.strong_best_record = [0 for x in range(self.device_num)]
        self.strong_second_record = [0 for x in range(self.device_num)]
        self.__arm_index_sum__ = list(np.cumsum(self.arm_nums))

    def next_pull(self):
        result = self.next_strong_pull()
        if result:
            return result
        else:
            self.T += 1
            return self.next_weak_pull()

    def next_weak_pull(self):
        arms = []
        for i in range(self.device_num):
            device = self.strong_pull_record[i]
            arm = max(device, key=itemgetter(0))
            index = device.index(arm)
            arms.append(index)
        return "weak", tuple(arms)

    def next_strong_pull(self):
        """
        Using modified methods from V Gabillon, etc., Multi-Bandit Best Arm Identification, 2011
        UCB_Complexity(device, arm) = (device.best_arm_mean - arm.mean) + sqrt(1/(2*arm.times)
        H = Sum(UCB_Complexity for (device,arm) in all arms)
        """
        def compute_H(device, arm_info):
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            complexity = diff + math.sqrt(1 / (2 * times))
            return 1 / float(complexity * complexity)

        def compute_B(device, a, arm_info):
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            score = -diff + math.sqrt(a / times)
            return score

        self.B += 1
        # out of budgets
        if self.B > self.strong_rounds:
            return False
        K = self.strong_arm_nums
        # initialization of pulls
        if self.B <= K:
            device = bisect.bisect(self.__arm_index_sum__, self.B-1)
            if device == 0:
                arm = self.B - 1
            else:
                arm = self.B - self.__arm_index_sum__[device-1] - 1
            return "strong", device, arm
        # using modified algorithms now
        n = self.strong_rounds
        H = 0
        for i in range(self.device_num):
            H += sum([compute_H(i, a) for a in self.strong_pull_record[i]])
        a = (4 / 9) * (n - K) / H
        candidates = []
        for i in range(self.device_num):
            scores = [compute_B(i, a, x) for x in self.strong_pull_record[i]]
            best_score = max(scores)
            index = scores.index(best_score)
            candidates.append((best_score, index))
        best = max(candidates, key=itemgetter(0))
        device = candidates.index(best)
        arm = best[1]
        return "strong", device, arm

    def update_strong_pull_record(self, device, arm, reward):
        arms = self.strong_pull_record[device]
        mean, times = arms[arm]
        mean = (mean * times + reward) / float(times + 1)
        self.strong_pull_record[device][arm] = (mean, times+1)
        best_arm = max(arms, key=itemgetter(0))
        arms_copy = arms[:]
        arms_copy.remove(best_arm)
        second_arm = max(arms_copy, key=itemgetter(0))
        self.strong_best_record[device] = best_arm[0]
        self.strong_second_record[device] = second_arm[0]


class StrongModelWeakPullAlgo(WeakStrongBanditAlgo):
    def initialize(self):
        super(StrongModelWeakPullAlgo, self).initialize()
        self.algo_name = "StrongModelWeakPullAlgo"
        self.strong_arm_nums = sum(self.arm_nums)
        self.weak_base = [0 for x in range(self.device_num)]
        extra_arm_nums = [x-1 for x in self.arm_nums]
        self.__initialized_index__ = list(np.cumsum(extra_arm_nums))
        self.model_pull_record = []
        for i in range(self.device_num):
            r = [(0, 0) for x in range(self.arm_nums[i])]
            # if i != 0:
            r[0] = (0, 1)
            self.model_pull_record.append(r)

    def next_pull(self):
        self.T += 1
        return self.next_weak_pull()

    def next_weak_pull(self):
        def compute_UCB(device, arm):
            mean, times = self.model_pull_record[device][arm]
            # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
            confidence = math.sqrt(2*math.log(self.T)/float(times))
            return mean + confidence
        K = self.strong_arm_nums
        n = self.device_num
        arms = [0 for x in range(self.device_num)]
        # initialization of pulls
        if self.T <= K - self.device_num + 1:
            device, arm = self.compute_device_arm_for_initialization()
            for i in range(device):
                arms[i] = self.model_pull_record[i].index(max(self.model_pull_record[i], key=itemgetter(0)))
            arms[device] = arm
        else:
            # using simultaneous UCB MAB algo
            for i in range(self.device_num):
                scores = [compute_UCB(i, a) for a in range(self.arm_nums[i])]
                # print("before computing the record is ", self.model_pull_record)
                # print("scores for device{0} is {1}".format(i, scores))
                arms[i] = scores.index(max(scores))
        return "weak", tuple(arms)

    def compute_device_arm_for_initialization(self):
        device = bisect.bisect(self.__initialized_index__, self.T - 2)
        if device == 0:
            arm = self.T - 1
        else:
            arm = self.T - self.__initialized_index__[device - 1] - 1
        return device, arm

    def update_weak_pull_record(self, arms, reward):
        K = self.strong_arm_nums
        n = self.device_num
        base_arms = self.weak_base
        # For initialization, we just fill in the first estimated diff between chosen arm and basis.
        if self.T == 1:
            True
        elif self.T <= K - n + 1:
            device, arm = self.compute_device_arm_for_initialization()
            base_val = self.weak_pull_record[tuple(base_arms)][0]
            self.adjust_model_pull_record(device, arms[device], (reward-base_val)/n, times_change=1)
        # After that, we start complicated updating process to maintain a strong_pull model.
        else:
            expected_diff_reward = [0 for i in range(n)]
            chosen_arms_times = [0 for i in range(n)]
            for i in range(n):
                arm_index = arms[i]
                arm_val_by_diff, times = self.model_pull_record[i][arm_index]
                expected_diff_reward[i] = arm_val_by_diff
                chosen_arms_times[i] = times
            base_val = self.weak_pull_record[tuple(base_arms)][0]
            diff = reward - sum(expected_diff_reward) - base_val
            for i in range(n):
                # base_index = base_arms[i]
                # arm_index, chosen_arm_times = , chosen_arms_times[i]
                self.update_difference(i, arms[i], diff, chosen_arms_times)
                # if base_index == arm_index:  # UNDECIDED: now it means the base arm doesn't update
                #     self.adjust_model_pull_record(i, arm_index, 0, times_change=1)
                # else:
                #     base_val, base_times = self.model_pull_record[i][base_index]
                #     arm_diff = diff_per_deivce * (base_times / base_times + chosen_arm_times)
                #     self.adjust_model_pull_record(i, arm_index, arm_diff, times_change=1)

                # base_diff = diff_per_deivce * (chosen_arm_times / chosen_arm_times + base_times)
                # arm_diff = diff_per_deivce - base_diff
                # for j in range(self.arm_nums[i]):
                #     if j == arm_index:
                #         self.adjust_model_pull_record(i, j, -arm_diff, times_change=1)
                #     else:
                #         self.adjust_model_pull_record(i, j, base_diff)
                # self.model_pull_record[i][arm_index] = (expected_diff_reward[i]+arm_diff, chosen_arms_times[i]+1)
            self.update_new_base()
        super(StrongModelWeakPullAlgo, self).update_weak_pull_record(arms, reward)

    def update_difference(self, device, arm, diff, times):
        n = self.device_num
        base_arms = self.weak_base
        diff = diff / n  # Average var by the number of devices.
        base_index = base_arms[device]
        chosen_arm_times = times[device]
        if base_index == arm:  # UNDECIDED: now it means the base arm doesn't update
            self.adjust_model_pull_record(device, arm, 0, times_change=1)
        else:
            base_val, base_times = self.model_pull_record[device][base_index]
            arm_diff = diff * (base_times / base_times + chosen_arm_times)
            self.adjust_model_pull_record(device, arm, arm_diff, times_change=1)

    def adjust_model_pull_record(self, device, arm, diff, times_change=1):
        assert times_change >= 0
        val, times = self.model_pull_record[device][arm]
        times = times + times_change
        if times != 0:
            if val + diff > 1:
                diff = 1 - val
            elif val + diff < -1:
                diff = 0
            val = (val * times + diff) / float(times)
            self.model_pull_record[device][arm] = (val, times)
            # print("adjusting device{0} arm{1}, the new record is {2}".format(device, arm, self.model_pull_record))

    def update_new_base(self, n=None):
        # print("try to update new base")
        if n is None:
            n = self.device_num
        for device in range(n):
            arms_record = self.model_pull_record[device]
            most_explored = max(arms_record, key=itemgetter(1))
            base_index = arms_record.index(most_explored)
            self.weak_base[device] = base_index
            for i in range(self.arm_nums[device]):
                diff = most_explored[0]
                self.adjust_model_pull_record(device, i, -diff, times_change=0)


class OptimisticStrongModelWeakPullAlgo(StrongModelWeakPullAlgo):
    def initialize(self):
        super(OptimisticStrongModelWeakPullAlgo, self).initialize()
        self.algo_name = "OptimisticStrongModelWeakPullAlgo"

    def next_weak_pull(self):
        def compute_UCB(device, arm):
            mean, times = self.model_pull_record[device][arm]
            # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
            confidence = math.sqrt(2 * math.log(self.T) / float(times))
            return mean + confidence

        K = self.strong_arm_nums
        n = self.device_num
        arms = [0 for x in range(self.device_num)]
        # initialization of pulls
        if self.T <= K - self.device_num + 1:
            device, arm = self.compute_device_arm_for_initialization()
            for i in range(device):
                arms[i] = self.model_pull_record[i].index(max(self.model_pull_record[i], key=itemgetter(0)))
            arms[device] = arm
        else:
            # using simultaneous UCB MAB algo
            for i in range(self.device_num):
                scores = [compute_UCB(i, a) for a in range(self.arm_nums[i])]
                arms[i] = scores.index(max(scores))
        return "weak", tuple(arms)

    def update_difference(self, device, arm, diff, times):
        n = self.device_num
        base_arms = self.weak_base
        diff = diff  # Average var by the number of devices.
        base_index = base_arms[device]
        chosen_arm_times = times[device]
        if base_index == arm:  # UNDECIDED: now it means the base arm doesn't update
            self.adjust_model_pull_record(device, arm, 0, times_change=1)
        else:
            base_val, base_times = self.model_pull_record[device][base_index]
            arm_diff = diff * (base_times / base_times + chosen_arm_times)
            self.adjust_model_pull_record(device, arm, arm_diff, times_change=1)


class RandomStrongModelWeakPullAlgo(StrongModelWeakPullAlgo):
    def initialize(self):
        super(RandomStrongModelWeakPullAlgo, self).initialize()
        self.algo_name = "RandomStrongModelWeakPullAlgo"

    def update_difference(self, device, arm, diff, times):
        n = self.device_num
        base_arms = self.weak_base
        diff = diff * random.randint(0, n) / n  # Average var by the number of devices.
        base_index = base_arms[device]
        chosen_arm_times = times[arm]
        if base_index == arm:  # UNDECIDED: now it means the base arm doesn't update
            self.adjust_model_pull_record(device, arm, 0, times_change=1)
        else:
            base_val, base_times = self.model_pull_record[device][base_index]
            arm_diff = diff * (base_times / base_times + chosen_arm_times)
            self.adjust_model_pull_record(device, arm, arm_diff, times_change=1)


class RespondingStrongModelWeakPullAlgo(StrongModelWeakPullAlgo):
    def initialize(self):
        super(RespondingStrongModelWeakPullAlgo, self).initialize()
        self.algo_name = "RespondingStrongModelWeakPullAlgo"

    def update_difference(self, device, arm, diff, times):
        n = self.device_num
        base_arms = self.weak_base
        chosen_arm_times = times[device]
        s = sum(times)
        if s > 0:
            diff = diff * chosen_arm_times / s  # Average var by the number of devices.
        base_index = base_arms[device]
        if base_index == arm:  # UNDECIDED: now it means the base arm doesn't update
            self.adjust_model_pull_record(device, arm, 0, times_change=1)
        else:
            base_val, base_times = self.model_pull_record[device][base_index]
            arm_diff = diff * (base_times / base_times + chosen_arm_times)
            self.adjust_model_pull_record(device, arm, arm_diff, times_change=1)


class StrongModelWeakStrongAlgo(StrongModelWeakPullAlgo, PureStrongPullAlgo):
    def initialize(self):
        super(StrongModelWeakStrongAlgo, self).initialize()
        self.algo_name = "StrongModelWeakStrongAlgo"
        self.strong_arm_nums = sum(self.arm_nums)
        self.strong_B = 0
        self.strong_best_record = [0 for x in range(self.device_num)]
        self.strong_second_record = [0 for x in range(self.device_num)]
        self.__arm_index_sum__ = list(np.cumsum(self.arm_nums))
        self.model_pull_record = []
        for i in range(self.device_num):
            r = [(0, 0) for x in range(self.arm_nums[i])]
            self.model_pull_record.append(r)

    def next_pull(self):
        if self.B < self.device_num:
            result = self.next_strong_pull()
            if result:
                return result
            else:
                self.T += 1
                return self.next_weak_pull()
        # TODO：asking for strong pulls according to some policy
        if self.B < self.strong_rounds:
            return self.next_strong_pull()
        else:
            self.T += 1
            return self.next_weak_pull()

    def next_strong_pull(self):
        def compute_H(device, arm_info):
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            complexity = diff + math.sqrt(1 / (2 * times))
            return 1 / float(complexity * complexity)

        def compute_B(device, a, arm_info):
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            score = -diff + math.sqrt(a / times)
            return score

        self.B += 1
        # out of budgets
        if self.B > self.strong_rounds:
            return False
        K = self.strong_arm_nums
        # initialization of pulls
        if self.B <= self.device_num:
            device = self.B - 1
            arm = 0
            return "strong", device, arm
        result = self.has_strong_arm_unexplored()
        if result:
            device, arm = result
        # if self.B <= K:
        #     device = bisect.bisect(self.__arm_index_sum__, self.B-1)
        #     if device == 0:
        #         arm = self.B - 1
        #     else:
        #         arm = self.B - self.__arm_index_sum__[device-1] - 1
            return "strong", device, arm
        # using modified algorithms now
        n = self.strong_rounds
        H = 0
        for i in range(self.device_num):
            H += sum([compute_H(i, a) for a in self.strong_pull_record[i]])
        a = (4 / 9) * (n - K) / H
        candidates = []
        for i in range(self.device_num):
            scores = [compute_B(i, a, x) for x in self.strong_pull_record[i]]
            best_score = max(scores)
            index = scores.index(best_score)
            candidates.append((best_score, index))
        best = max(candidates, key=itemgetter(0))
        device = candidates.index(best)
        arm = best[1]
        return "strong", device, arm

    def next_weak_pull(self):
        def compute_UCB(device, arm):
            mean, times = self.model_pull_record[device][arm]
            # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
            confidence = math.sqrt(2*math.log(self.T)/float(times))
            return mean + confidence
        K = self.strong_arm_nums
        n = self.device_num
        arms = [0 for x in range(self.device_num)]
        # initialization of pulls
        result = self.has_strong_arm_model_unexplored()
        if result:
            device, arm = result
            for i in range(device):
                arms[i] = self.model_pull_record[i].index(max(self.model_pull_record[i], key=itemgetter(0)))
            arms[device] = arm
        else:
            # using simultaneous UCB MAB algo
            for i in range(self.device_num):
                scores = [compute_UCB(i, a) for a in range(self.arm_nums[i])]
                # print(self.model_pull_record)
                # print("scores for device{0} is {1}".format(i, scores))
                arms[i] = scores.index(max(scores))
        return "weak", tuple(arms)

    def has_strong_arm_unexplored(self):
        for i in range(self.device_num):
            for j in range(self.arm_nums[i]):
                if self.strong_pull_record[i][j][1] == 0:
                    return i, j
        return False

    def has_strong_arm_model_unexplored(self, arms=None):
        for i in range(self.device_num):
            if arms is None:
                for j in range(self.arm_nums[i]):
                    if self.model_pull_record[i][j][1] == 0:
                        return i, j
            else:
                arm = arms[i]
                if self.model_pull_record[i][arm] == 0:
                    return i, arm
        return False

    def update_strong_pull_record(self, device, arm, reward):
        # TODO: model adjustment caused by strong_pull may have more weight.
        self.update_strong_pull_without_model_record(device, arm, reward)
        base_index = self.weak_base[device]
        base_mean, base_times = self.strong_pull_record[device][base_index]
        if base_times > 0:
            self.adjust_model_pull_record(device, arm, reward-base_mean)
            self.update_new_base(n=device)

    def update_strong_pull_without_model_record(self, device, arm, reward):
        arms = self.strong_pull_record[device]
        mean, times = arms[arm]
        mean = (mean * times + reward) / float(times + 1)
        self.strong_pull_record[device][arm] = (mean, times + 1)
        best_arm = max(arms, key=itemgetter(0))
        arms_copy = arms[:]
        arms_copy.remove(best_arm)
        second_arm = max(arms_copy, key=itemgetter(0))
        self.strong_best_record[device] = best_arm[0]
        self.strong_second_record[device] = second_arm[0]

    def update_weak_pull_record(self, arms, reward):
        K = self.strong_arm_nums
        n = self.device_num
        base_arms = self.weak_base
        # For initialization, we just fill in the first estimated diff between chosen arm and basis.
        result = self.has_strong_arm_model_unexplored(arms=arms)
        if result:
            device, arm = result
            base_val = self.weak_pull_record[tuple(base_arms)][0]
            diff = (reward - base_val) / n
            self.adjust_model_pull_record(device, arms[device], (reward - base_val) / n, times_change=1)
            base_index = base_arms[device]
            base_mean, base_times = self.strong_pull_record[device][base_index]
            if base_times > 0:
                self.update_strong_pull_without_model_record(device, arm, base_mean + diff)
        # After that, we start complicated updating process to maintain a strong_pull model.
        else:
            # print("strong record ", self.strong_pull_record)
            # print("model record ", self.model_pull_record)
            expected_diff_reward = [0 for i in range(n)]
            chosen_arms_times = [0 for i in range(n)]
            for i in range(n):
                arm_index = arms[i]
                arm_val_by_diff, times = self.model_pull_record[i][arm_index]
                expected_diff_reward[i] = arm_val_by_diff
                chosen_arms_times[i] = times
            base_val = self.weak_pull_record[tuple(base_arms)][0]
            diff = reward - sum(expected_diff_reward) - base_val
            for i in range(n):
                self.update_difference(i, arms[i], diff, chosen_arms_times)
            self.update_new_base()
        super(StrongModelWeakPullAlgo, self).update_weak_pull_record(arms, reward)

    def update_difference(self, device, arm, diff, times):
        n = self.device_num
        base_arms = self.weak_base
        diff = diff / n  # Average var by the number of devices.
        chosen_arm_times = times[device]
        base_index = base_arms[device]
        base_mean, base_times = self.strong_pull_record[device][base_index]
        if base_times > 0:
            self.update_strong_pull_without_model_record(device, arm, base_mean + diff)
        if base_index == arm:  # UNDECIDED: now it means the base arm doesn't update
            self.adjust_model_pull_record(device, arm, 0, times_change=1)
        else:
            base_val, base_times = self.model_pull_record[device][base_index]
            arm_diff = diff * (base_times / base_times + chosen_arm_times)
            self.adjust_model_pull_record(device, arm, arm_diff, times_change=1)

    def update_new_base(self, n=None):
        if n is None:
            n = self.device_num
        for device in range(n):
            arms_record = self.model_pull_record[device]
            most_explored = max(arms_record, key=itemgetter(1))
            base_index = arms_record.index(most_explored)
            self.weak_base[device] = base_index
            for i in range(self.arm_nums[device]):
                diff = most_explored[0]
                self.adjust_model_pull_record(device, i, -diff, times_change=0)


class SingleMABWithStrongPull(WeakStrongBanditAlgo):
    """
    Only considers one bandit. Every K rounds it can incur a strong pull freely.
    Weak pulls are in fact stored in a strong pull record, considering strong_pull_record is a list while weak's is dict
    """
    def __init__(self, arm_nums, pulls, rounds, budgets, algo=None, k=None, complexity=None):
        super(SingleMABWithStrongPull, self).__init__(arm_nums, pulls, rounds, budgets, algo)
        if k is None or k <= 0:
            self.k = 0
            self.k_rounds = 0
        else:
            self.k = k
            self.k_rounds = rounds // k
        self.complexity = False if complexity is None else complexity
        self.algo_name = "SingleMABWithStrongPull-{0}".format(self.k)
        self.T = 1
        self.is_strong_pulled = True

    def initialize(self):
        super(SingleMABWithStrongPull, self).initialize()
        self.arm_num = self.arm_nums[0]
        self.strong_best_record = [0 for x in range(self.device_num)]
        self.strong_second_record = [0 for x in range(self.device_num)]

    def next_pull(self):
        self.T += 1
        if not self.is_strong_pulled and self.k != 0 and (self.T-1) % self.k == 0:
            self.T -= 1
            self.is_strong_pulled = True
            return self.next_strong_pull()
        else:
            self.is_strong_pulled = False
            return self.next_weak_pull()

    def next_strong_pull(self):
        def compute_H(device, arm_info):
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            complexity = diff + math.sqrt(1 / (2 * times))
            return 1 / float(complexity * complexity)

        def compute_B(device, a, arm_info):
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            score = -diff + math.sqrt(a / times)
            return score
        n = self.arm_num
        rounds = self.rounds + self.k_rounds
        # initialization of pulls
        result = self.has_arm_unexplored()
        device = 0
        if result is 0 or result:
            arm = result
        else:
            if self.complexity:
                H = self.complexity
            else:
                H = 0
                for i in range(self.device_num):
                    H += sum([compute_H(i, a) for a in self.strong_pull_record[i]])
            a = (25 / 36) * (rounds - n) / H
            candidates = []
            for i in range(self.device_num):
                scores = [compute_B(i, a, x) for x in self.strong_pull_record[i]]
                best_score = max(scores)
                index = scores.index(best_score)
                candidates.append((best_score, index))
            best = max(candidates, key=itemgetter(0))
            device = candidates.index(best)
            arm = best[1]
        return "strong", device, arm

    def has_arm_unexplored(self):
        for i in range(self.arm_num):
            mean, times = self.strong_pull_record[0][i]
            if times == 0:
                return i
        return False

    def next_weak_pull(self):
        def compute_UCB(device, arm):
            mean, times = self.strong_pull_record[device][arm]
            # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
            confidence = math.sqrt(2*math.log(self.T)/float(times))
            return mean + confidence
        n = self.arm_num
        # initialization of pulls
        result = self.has_arm_unexplored()
        if result is 0 or result:
            arm = result
        else:
            scores = [compute_UCB(0, a) for a in range(n)]
            arm = scores.index(max(scores))
        return "weak", tuple([arm])

    def update_weak_pull_record(self, arms, reward):
        self.update_strong_pull_record(0, arms[0], reward)

    def update_strong_pull_record(self, device, arm, reward):
        arms = self.strong_pull_record[device]
        mean, times = arms[arm]
        mean = (mean * times + reward) / float(times + 1)
        self.strong_pull_record[device][arm] = (mean, times+1)
        best_arm = max(arms, key=itemgetter(0))
        arms_copy = arms[:]
        arms_copy.remove(best_arm)
        second_arm = max(arms_copy, key=itemgetter(0))
        self.strong_best_record[device] = best_arm[0]
        self.strong_second_record[device] = second_arm[0]


class SimulatorWithK(Simulator):
    def __init__(self, devices, pulls, rounds, budgets, algos=None, generation=None, reward_func=None,
                 strong_pull_func=None, weak_pull_func=None, k=None, complexity=None):
        # if not isinstance(devices, MultiDevices):
        #     raise TypeError("The first argument should be a MultiDevices object.")
        if not self.__devices_check__(devices):
            raise TypeError("The first argument should be a list of lists of floats between 0 and 1")
        if (not isinstance(pulls, dict)) or ("weak" not in pulls.keys()) or ("strong" not in pulls.keys()):
            raise TypeError("pulls should be a dictionary containing 'strong' and 'weak' keys.")

        def reward_generator(reward):
            st_dev = 0.1
            r = np.random.normal(reward, st_dev) if generation is None else generation(reward)
            if r > 1:
                r = 1
            if r < 0:
                r = 0
            return r

        self.devices = MultiDevices(devices, reward_generator, reward_func)
        self.pulls = pulls
        self.rounds = rounds
        self.budgets = budgets
        arm_nums = self.devices.get_arm_nums()
        self.algos = []
        if algos is None:
            self.algos = [WeakStrongBanditAlgo(arm_nums, pulls, rounds, budgets)]
        elif isinstance(algos, list):
            for algo in algos:
                self.algos.append(algo(arm_nums, pulls, rounds, budgets))
        elif algos == SingleMABWithStrongPull:
            for i in range(k+1):
                self.algos.append(algos(arm_nums, pulls, rounds, budgets, k=i, complexity=complexity))
        else:
            self.algos = [algos(arm_nums, pulls, rounds, budgets)]
        self.strong_pull_func = strong_pull_func
        self.weak_pull_func = weak_pull_func


if __name__ == "__main__":
    def reward_generator(reward, st_dev=0.2):
        st_dev = 0
        r = np.random.normal(reward, st_dev)
        if r > 1:
            r = 1
        if r < 0:
            r = 0
        return r

    def compute_true_H(devices):
        def compute_H_for_single_bandit(arm, best, second):
            diff = best - arm if arm - best != 0 else best - second
            # complexity = diff + math.sqrt(1 / (2 * times))
            complexity = diff
            return 1 / float(complexity * complexity)
        H = 0
        for device in devices:
            best = max(device)
            copy = device[:]
            copy.remove(best)
            second = max(copy)
            H += sum([compute_H_for_single_bandit(a, best, second) for a in device])
        return H


    devices = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    devices = [[0.3, 0.4, 0.5, 0.6, 0.7]]
    pulls = {"strong": 1, "weak": 1}
    rounds = 70
    budgets = 11
    # devices_obj = MultiDevices(devices, reward_generator)
    # arm_num = devices_obj.get_arm_nums()
    # algos = [CombinatorialWeakPullAlgo, StrongModelWeakPullAlgo, StrongModelWeakStrongAlgo, PureStrongPullAlgo]
    algos = SingleMABWithStrongPull
    simu = SimulatorWithK(devices, pulls, rounds, budgets, algos=algos, generation=reward_generator, k=3, complexity=compute_true_H(devices))
    simu.quick_run(silent=False)

    # simu = Simulator(devices, pulls, rounds, budgets, algos=algos)
    # simu = SimulatorWithK(devices, pulls, rounds, budgets, algos=algos, k=10)

    # results = []
    # k = [x for x in range(11)]
    # max_rounds = 110
    # for rounds in range(10, max_rounds, 10):
    #     simu = SimulatorWithK(devices, pulls, rounds, budgets, algos=algos, k=10)
    #     results_of_run = simu.run(150, True)
    #     results.append(results_of_run)
    #     plt.figure()
    #     plt.bar(k, results_of_run)
    #     plt.xlabel("k")
    #     plt.ylabel("Average regrets of 170 trials")
    #     plt.title("Pulls = {0}".format(rounds))
    #     plt.show()


