import numpy as np
# import math
# import bisect
# import random


# Pragma: Basic classes

class Device(object):
    """
    A Device object is a normal MAB setting, meaning a limited number of arms with true rewards and some reward
    generating function.
    """

    def __init__(self, arms, st_dev=-1):
        """
        Args:
            arms: List[float/Tuple]. DecraptDict[str, float/Tuple]. Key is arm's name, value is other info
        """
        def binary_generator(r):
            return np.random.choice([0, 1], p=[1 - r, r])
        def gaussian_generator(reward):
            r = np.random.normal(reward, st_dev)
            if r > 1:
                r = 1
            if r < 0:
                r = 0
            return r
        self.arms = arms
        self.best_arm_val = max(arms)
        self.best_arm_index = arms.index(self.best_arm_val)
        self.generator = gaussian_generator if st_dev >= 0 else binary_generator

    def pull(self, arm):
        """
        Args:
            arm: the arm's name/identifier
        """
        return self.generator(self.arms[arm])

    def get_best_arm(self):
        """
        Returns (the best arm index, its true reward)
        """
        return self.best_arm_index, self.best_arm_val

    def get_true_reward(self, arm):
        return self.arms[arm]

    def get_arm_num(self):
        return len(self.arms)


class Simulator(object):
    def __init__(self, true_rewards, experiment_options):
        # if not isinstance(devices, MultiDevices):
        #     raise TypeError("The first argument should be a MultiDevices object.")

        # def reward_generator(reward):
        #     st_dev = 0.1 * multiplier
        #     r = np.random.normal(reward, st_dev)
        #     if r > multiplier:
        #         r = multiplier
        #     if r < 0:
        #         r = 0
        #     return r
        # if generator is None:
        #     generator = reward_generator

        # check input
        for arm in true_rewards:
            assert 0 <= arm <= 1
        self.device = Device(true_rewards, experiment_options["st_dev"])
        # decode options
        self.rounds = experiment_options["rounds"]
        self.k = experiment_options["k"]
        self.trials = experiment_options["trials"]
        self.interval = experiment_options["interval"]

    def run(self, Algo, free_policy=None, log_options=None):
        """
        Run experiments to get the regret difference between k=1 and k=0 (intervaled)

        """
        print("called once")
        times = self.trials
        interval = self.interval
        if log_options is None:
            silent, log_pulling_times, return_diff, avg_regret = True, False, True, False
        else:
            silent, log_pulling_times, return_diff, avg_regret, persistence = \
                log_options["silent"], log_options["log_pulling_times"], \
                log_options["return_diff"], log_options["avg_regret"], \
                log_options["persistence"]
        all_intervaled_regrets = []
        all_pulling_times = []
        k_arr = [1, 0]
        # run algo with free_policy on different k
        for k in k_arr:
            algo = Algo(self.device, self.rounds, k, free_policy)
            intervaled_regrets, intervaled_pulling_times = algo.run(times, interval, silent, log_pulling_times, avg_regret)
            all_intervaled_regrets.append(intervaled_regrets)
            if log_pulling_times:
                all_pulling_times.append(intervaled_pulling_times)
        regrets_diff = all_intervaled_regrets[0] - all_intervaled_regrets[-1]
        regrets_diff = np.round(regrets_diff, 3).tolist()
        # all_intervaled_regrets = np.array(all_intervaled_regrets).transpose().tolist()
        all_intervaled_regrets = np.array(all_intervaled_regrets).tolist()
        if log_pulling_times:
            all_pulling_times = np.array(all_pulling_times).transpose((1, 0, 2)).tolist()
        # start to output
        if interval is None:
            interval = self.rounds
        x_axis_length = int((self.rounds - 0.5) // interval) + 1
        for i in range(x_axis_length):
            print(f"after {min((i + 1) * interval, self.rounds)} round")
            # print("the average regrets are ", all_intervaled_regrets[i])
            print(f"Max free pull changes regrets: {regrets_diff[i]}")
            if log_pulling_times:
                print("the number of pulls for arms are :")
                for k, record in zip(k_arr, all_pulling_times[i]):
                    print(f"k={k} pulls: {record}")
            print("")
        if return_diff:
            return regrets_diff, all_pulling_times
        return all_intervaled_regrets, all_pulling_times


class FreePullBandit(object):
    """
    Attributes:
        arm_num: int. The number of arms
        rounds: same as in Simulator
        k: free pull frequency
        free_policy:

        T: current round
        free_pull_record: List[List[float]]. records = strong_pull_record[device][arm]
        normal_pull_record: Dict[tuple: List[float]]. records = weak_pull_record[(d1_a,d2_a,...)]

    Methods:
        run_once() and run() will return np.array instead of list, because it can save time to convert them
        if returning values will be used for further computation
    """

    def __init__(self, device, rounds, k, free_policy=None):
        def random_pull(self):
            return np.random.randint(0, self.arm_num)
        self.device = device
        self.arm_num = device.get_arm_num()
        self.k = k
        self.rounds = rounds
        self.free_policy = free_policy
        self.free_pull = free_policy if free_policy is not None else random_pull
        self.reinitialize()
        # self.T = 0
        # self.normal_pull_record = [(0, 0) for x in range(self.arm_num)]
        # self.free_pull_record = self.normal_pull_record[:]

    def reinitialize(self):
        self.T = 0
        self.normal_pull_record = [(0, 0) for x in range(self.arm_num)]
        self.free_pull_record = self.normal_pull_record[:]
        self.a = np.ones(self.arm_num)
        self.b = np.ones(self.arm_num)

    def pull_and_update_record(self, arm, is_free_pull=False):
        reward = self.device.pull(arm)
        # update (mean, times) record for potential free pulls
        # and for logging pulling times
        mean, times = self.normal_pull_record[arm]
        mean = (mean * times + reward) / float(times + 1)
        self.normal_pull_record[arm] = (mean, times + 1)
        if is_free_pull:
            mean, times = self.free_pull_record[arm]
            mean = (mean * times + reward) / float(times + 1)
            self.free_pull_record[arm] = (mean, times + 1)
        # update posterior beta distribution
        if reward == 1:
            self.a[arm] += 1
        else:
            self.b[arm] += 1
        return reward

    def normal_pull(self):
        return 0

    # def free_pull(self):
    #     return self.random_pull()
    #
    def random_pull(self):
        return np.random.randint(0, self.arm_num)

    def has_arm_unexplored(self):
        for i in range(self.arm_num):
            mean, times = self.normal_pull_record[i]
            if times == 0:
                return i
        return False

    def run_once(self, interval=None, silent=True, log_pulling_times=False, avg_regret=False):
        self.reinitialize()
        total_reward = 0
        best_reward = self.device.get_best_arm()[1]

        intervaled_regret = []
        intervaled_pulling_times = []
        if not silent:
            print("The best reward is {0}".format(best_reward))
        for i in range(1, self.rounds + 1):
            self.T = i
            # do normal pulls
            arm = self.normal_pull()
            reward = self.pull_and_update_record(arm, is_free_pull=False)
            if not silent:
                print("round {0}, choose arm {1} and get reward {2}".format(i, arm, reward))
            # do free pulls
            if self.k != 0 and (i - 1) % self.k == 0:
                arm = self.free_pull(self)
                self.pull_and_update_record(arm, is_free_pull=True)
            # update rewards
            total_reward += reward
            # handle about intervals
            if interval is not None and i % interval == 0:
                if avg_regret:
                    intervaled_regret.append(best_reward - total_reward / i)
                else:
                    intervaled_regret.append(best_reward * i - total_reward)
                if log_pulling_times:
                    pulling_times = [x[1]-y[1] for x,y in zip(self.normal_pull_record, self.free_pull_record)]
                    intervaled_pulling_times.append(pulling_times)
        # if interval isn't set or rounds is not a multiple of interval
        if interval is None or self.rounds % interval != 0:
            # regret = best_reward * self.rounds - total_reward
            if avg_regret:
                intervaled_regret.append(best_reward - total_reward / i)
            else:
                intervaled_regret.append(best_reward * i - total_reward)
            if log_pulling_times:
                pulling_times = [x[1] for x in self.normal_pull_record]
                intervaled_pulling_times.append(pulling_times)
        if not silent:
            print("after {0} rounds, the total regret is {1}".format(self.rounds, intervaled_regret[-1]))
        return np.array(intervaled_regret), np.array(intervaled_pulling_times)

    def run(self, times, interval=None, silent=True, log_pulling_times=False, avg_regret=False):
        if interval is None:
            interval = self.rounds

        x_axis_length = int((self.rounds - 0.5) // interval) + 1
        sum_intervaled_regrets = np.zeros(x_axis_length)
        if log_pulling_times:
            sum_intervaled_pulling_times = np.zeros([x_axis_length, self.arm_num])
        for i in range(times):
            intervaled_regrets, intervaled_pulling_times = self.run_once(interval, silent, log_pulling_times, avg_regret)
            sum_intervaled_regrets += np.array(intervaled_regrets)
            if log_pulling_times:
                sum_intervaled_pulling_times += intervaled_pulling_times
        average_intervaled_regrets = np.round(sum_intervaled_regrets / times, 3)
        intervaled_pulling_times = sum_intervaled_pulling_times/times if log_pulling_times else []
        return np.round(sum_intervaled_regrets/times, 3), intervaled_pulling_times
