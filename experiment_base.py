import numpy as np
from utils import *


class Device(object):
    """
    A Device object is a normal MAB setting, meaning a limited number of arms with true rewards and some reward
    generating function.
    """

    def __init__(self, arms, st_dev=None):
        """
        Args:
            arms: List[float/Tuple]. DecraptDict[str, float/Tuple]. Key is arm's name, value is other info
            st_dev: 
                float/None. The standard deviation for gaussion distribution. The device will use binary generator
                if this is None.
        """
        self.__arms = arms
        self.__best_arm_val = max(arms)
        self.__best_arm_index = arms.index(self.__best_arm_val)
        self.__generator = gaussian_sampler_generator(st_dev) if st_dev is not None else binary_sampler

    def pull(self, arm):
        """Pulls an arm from set generator.
        Args:
            arm: the arm's name/identifier
        """
        return self.__generator(self.__arms[arm])

    def get_best_arm(self):
        """Returns (the best arm index, its true reward)
        """
        return self.__best_arm_index, self.__best_arm_val

    def get_true_reward(self, arm):
        return self.__arms[arm]

    def get_arm_num(self):
        return len(self.__arms)


class Simulator(object):
    def __init__(self, device, experiment_options):
        """Initializes the instance based on experiment options.

        Args:
            device: Device.
            experiment_options: 
                st_dev: float/None. Standard deviation for gaussion distribution sampler. None for
                    binary distribution.
                rounds: int. Total number of pull rounds.
                k: non-negative int. Free pull frequency. O means no free pulls for comparison's sake
                trials: int. Number of ex
                interval: int
        """
        self.device = device
        
        # decode options
        self.rounds = experiment_options["rounds"]
        self.k = experiment_options["k"]
        self.trials = experiment_options["trials"]
        self.interval = experiment_options["interval"]

    def run(self, Algo, free_policy=None, log_options=None):
        """
        Run experiments to get the regret difference between k=1 and k=0 (intervaled)

        Args:
            Algo: FreePullBandit
            free_policy: (FreePullBandit) => int. free pull policy function
            log_options: 
                silent: boolean
                log_pulling_times: boolean
                return_diff: boolean
                avg_regret: boolean
                persistence: boolean
        """
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
        # generate k arr: [1, .., k, 0]
        k_arr = list(range(1, self.k+1))
        k_arr.append(0)

        # run algo with free_policy on different k
        for k in k_arr:
            algo = Algo(self.device, self.rounds, k, free_policy)
            intervaled_regrets, intervaled_pulling_times = algo.run(times, interval, silent, log_pulling_times, avg_regret)
            all_intervaled_regrets.append(intervaled_regrets)
            if log_pulling_times:
                all_pulling_times.append(intervaled_pulling_times)

        regrets_diff = all_intervaled_regrets[-1] - all_intervaled_regrets[0]
        regrets_diff = np.round(regrets_diff, 3).tolist()
        all_intervaled_regrets = np.array(all_intervaled_regrets).tolist()
        if log_pulling_times:
            all_pulling_times = np.array(all_pulling_times).transpose((1, 0, 2)).tolist()

        # start to output
        if interval is None:
            interval = self.rounds
        x_axis_length = int((self.rounds - 0.5) // interval) + 1
        for i in range(x_axis_length):
            # print(f"after {min((i + 1) * interval, self.rounds)} round")
            # print("the average regrets are ", all_intervaled_regrets[i])
            # print(f"Max free pull changes regrets: {regrets_diff[i]}")
            if log_pulling_times:
                print("the number of pulls for arms are :")
                for k, record in zip(k_arr, all_pulling_times[i]):
                    print(f"k={k} pulls: {record}")
        if return_diff:
            return regrets_diff, all_pulling_times
        return all_intervaled_regrets, all_pulling_times


class FreePullBandit(object):
    """
    Attributes:
        arm_num: int. The number of arms.
        rounds: int. Total number of pull rounds.
        k: non-negative int. Free pull frequency. O means no free pulls for comparison's sake
        free_policy: free pull policy function

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
        self.free_pull = free_policy if free_policy is not None else random_pull
        self.initialize()

    def initialize(self):
        self.T = 0

        # Record history in different ways.
        # TODO: maybe it could be decoupled to each subclass.
        # for UCB
        self.normal_pull_record = [(0, 0) for x in range(self.arm_num)]
        self.free_pull_record = self.normal_pull_record[:]

        # for Thompson sampling
        self.a = np.ones(self.arm_num)
        self.b = np.ones(self.arm_num)

        # for successive elimination 
        self.rejected = [0] * self.arm_num

    def __update_ucb_record(self, arm, reward, is_free_pull=False):
        record = self.normal_pull_record if not is_free_pull else self.free_pull_record
        mean, times = record[arm]
        mean = (mean * times + reward) / float(times + 1)
        record[arm] = (mean, times + 1)
    
    def __update_beta_record(self, arm, reward):
        # update posterior beta distribution
        if reward == 1:
            self.a[arm] += 1
        else:
            self.b[arm] += 1
    
    def pull_and_update_record(self, arm, is_free_pull=False):
        reward = self.device.pull(arm)

        self.__update_ucb_record(arm, reward, False)
        self.__update_beta_record(arm, reward)

        if is_free_pull:
            self.__update_ucb_record(arm, reward, True)

        return reward

    def normal_pull(self):
        pass

    def random_pull(self):
        return np.random.randint(0, self.arm_num)

    def get_unexplored_arm(self):
        """Returns index or None.
        """
        for i in range(self.arm_num):
            _, times = self.normal_pull_record[i]
            if times == 0:
                return i
        return None

    def run_once(self, interval=None, silent=True, log_pulling_times=False, avg_regret=False):
        """Run experiment once, meaning do normal pulls self.rounds times, and do free pulls 
        every k rounds.
        """
        self.initialize()
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
            # not self.get_unexplored_arm() and
            if self.k != 0 and i >= self.device.get_arm_num() and (i - 1) % self.k == 0:
                arm = self.free_pull(self)
                r = self.pull_and_update_record(arm, is_free_pull=True)
                if not silent:
                    print("choose arm {1} as free pull with reward {2}".format(i, arm, r))
            
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
        """Run self.run_once() times. Compute needed statistics.
        """
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
