from operator import itemgetter
import numpy as np


class Simulator(object):
    """
    Simulator is the whole class of the simulator.
    It will create a MultiDevices and a WeakStrongBanditAlgo objects

    Attributes:
        The simulator takes (devices, pulls, rounds, budget, algorithm) as inputs. The first four parameters mean an
        environment setting, or in other words, a problem to be soloved. algorithm is some program that selects an arm
        for each round.

        devices: List[List[float]].The true reward of arms of devices, which is between 0 and 1.

        pulls. Ditc{str:tuple}. A list of the types of pulls the bandits can operate. A pull should contain a cost value
        and other information the algorithm needs to know to decide how to choose among different kinds of pulls, such
        as information gain in the paper.
        
        rounds. int. The total number of rounds the algorithm can play.
        
        budget. int/float. The total number of budgets the algorithm can spend on arms to select.
        
        algo. Optional[Function]. A function able to replace next_pull() method of a WeakStrongBanditAlgo, able to read
        pulling history and to output.

        generation: Optional[Function]. A function taking true reward as input and outputing a stochastic reward.
        Default setting is a Gaussian randaom generator with standard deviation 0.1.

        reward_func: Optional[Function]. A function taking a list of rewards from all devices as input and outputting a
        total reward. Default is simple sum().

        strong_pull_func: Optional[Function]. A function similar to generation, but used in strong_pull.

        weak_pull_func: Optional[Function]. A function similar to generation, but used in weak_pull.
    """

    def __init__(self, devices, pulls, rounds, budgets, multiplier=1, algos=None, generation=None, reward_func=None,
                 strong_pull_func=None, weak_pull_func=None):
        # if not isinstance(devices, MultiDevices):
        #     raise TypeError("The first argument should be a MultiDevices object.")
        self.multiplier = multiplier
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
        if generation is not None:
            reward_generator = generation
        self.devices = MultiDevices(devices, reward_generator, reward_func)
        self.pulls = pulls
        self.rounds = rounds
        self.budgets = budgets
        arm_nums = self.devices.get_arm_nums()
        if algos is None:
            self.algos = [WeakStrongBanditAlgo(arm_nums, pulls, rounds, budgets)]
        elif isinstance(algos, list):
            self.algos = []
            for algo in algos:
                self.algos.append(algo(arm_nums, pulls, rounds, budgets))
        else:
            self.algos = [algos(arm_nums, pulls, rounds, budgets)]
        self.strong_pull_func = strong_pull_func
        self.weak_pull_func = weak_pull_func
    
    def quick_run(self, silent=False):
        return self.run(1, silent)

    def __run_algo__(self, algo, silent=False):
        algo.initialize()
        total_reward = 0
        response = None
        best_reward = self.devices.get_best_reward()
        if not silent:
            print("The best reward is {0}".format(best_reward))
        for i in range(1, self.rounds + 1):
            chosen_arm = algo.choose_new_arm(response)
            assert chosen_arm
            pull_type, *selection_info = chosen_arm
            while pull_type == "strong":
                device, arm = selection_info
                reward = self.devices.strong_pull(device, arm, self.strong_pull_func)
                response = (pull_type, device, arm, reward)
                if not silent:
                    print("round {0}, response is {1}".format(i, response))
                chosen_arm = algo.choose_new_arm(response)
                assert chosen_arm
                pull_type, *selection_info = chosen_arm
            if pull_type == "weak":
                arms = selection_info[0]
                reward = self.devices.weak_pull(arms, self.weak_pull_func)
                response = (pull_type, arms, reward)
            else:
                raise TypeError("Pull_type should be 'strong' or 'weak'.")
            if not silent:
                print("round {0}, response is {1}".format(i, response))
            total_reward += reward
        regret = best_reward * self.rounds - total_reward
        if not silent:
            print("after {0} rounds, the total regret is {1} for {2}".format(self.rounds, regret, algo.algo_name))
        return regret

    def run(self, times, silent=False):
        regrets = []
        algo_names = []
        for algo in self.algos:
            algo_regrets = []
            for i in range(times):
                r = self.__run_algo__(algo, silent)
                algo_regrets.append(r)
            average_regret = sum(algo_regrets)/times
            regrets.append(average_regret)
            algo_names.append(algo.algo_name)
        print(algo_names)
        print("the average regrets are ", regrets)
        return regrets

    def __devices_check__(self, devices):
        if (not devices) or (not isinstance(devices, list)):
            return False
        for device in devices:
            if not isinstance(device, list):
                return False
            for arm in device:
                if arm > 1*self.multiplier or arm < 0:
                    return False
        return True


class Device(object):
    """
    A Device object is a normal MAB setting, meaning a limited number of arms with true rewards and some reward
    generating function.
    """
    def __init__(self, arms, generation):
        """
        Args:
            arms: List[float/Tuple]. DecraptDict[str, float/Tuple]. Key is arm's name, value is other info
        """
        # self.__arms__ = list(info_of_arms.keys())
        # self.__best_arm__ = max(info_of_arms, key=info_of_arms.get)
        
        self.__arms_num__ = len(arms)
        self.__arms__ = arms
        # self.arm_info = "simple"
        if arms and type(arms[0]) is not float:
            self.__arm_info__ = "advanced" 
            best_arm_val = max(arms, key=itemgetter(1)) 
        else:
            self.__arm_info__ = "simple"
            best_arm_val = max(arms)
        self.__best_arm_val__ = best_arm_val
        self.__best_arm__ = arms.index(best_arm_val)
        self.__generation__ = generation
    
    def pull(self, arm, pull_func=None):
        """
        Args:
            arm: the arm's name/identifier
            pull_func: a function taking the true reward as input, deciding how a reward is received from the pull
        """
        true_reward = self.get_true_reward(arm)
        if pull_func is None:
            received_reward = self.__generation__(true_reward)
        else:
            received_reward = pull_func(true_reward)
        return received_reward

    def get_best_arm(self):
        """
        Returns (the best arm index, its true reward)
        """
        return self.__best_arm__, self.__best_arm_val__        

    def get_true_reward(self, arm):
        r = self.__arms__[arm]
        if self.__arm_info__ == "advanced":
            r = r[0]
        return r
    
    def get_arm_num(self):
        return self.__arms_num__


class MultiDevices(object):
    """
    It's a gang of bandits problem. An "arm" here means a combination of arms of all bandits.
    It has strong_pull and weak_pull methods, which cooperates argument "pulls" in Simulator and WeakStrongBanditAlgo.
    If more pulling types are needed, new classes of MulitDevices and BanditAlgo should be created.
    """
    def __init__(self, info_of_devices, generation, reward_func=None):
        devices = []
        arm_nums = []
        best_reward = 0
        for info in info_of_devices:
            device = Device(info, generation)
            devices.append(device)
            arm_nums.append(device.get_arm_num())
            best_arm_val = device.get_best_arm()[1]
            best_reward += best_arm_val
        self.__best_reward__ = best_reward
        self.__devices__ = devices
        self.__device_num__ = len(devices)
        self.__generation__ = generation
        self.__arm_nums__ = arm_nums
        self.__reward_func__ = reward_func
        
    def get_devices(self):
        return self.__devices__
    
    def get_arm_nums(self):
        return self.__arm_nums__

    def get_best_reward(self):
        return self.__best_reward__

    def weak_pull(self, arms, pull_func=None):
        rewards = []
        devices = self.__devices__
        reward_func = self.__reward_func__
        arms = list(arms)
        for i, arm in enumerate(arms):
            # print("device {0} with arm {1}".format(i, arm))
            device = devices[i]
            r = device.pull(arm, pull_func=pull_func)
            rewards.append(r)
        received_reward = sum(rewards) if reward_func is None else reward_func(rewards)
        return received_reward

    def strong_pull(self, device, arm, pull_func=None):
        device = self.__devices__[device]
        return device.pull(arm, pull_func=pull_func)

    def custom_pull(self, arms, device_pull_func=None, reward_func=None):
        """
        Args:
            arms: List[int]. Arm indices for all devices
            device_pull_func: it's a function that takes device index as input and outputs a pull_func function
            reward_func: take a list of returning values of device.pull(arm) as input 
        """
        rewards = []
        devices = self.__devices__
        for i, arm in enumerate(arms):
            device = devices[i]
            pull_func = None if device_pull_func is None else device_pull_func(i)
            r = device.pull(arm, pull_func=pull_func)
            rewards.append(r)
        received_reward = sum(rewards) if reward_func is None else reward_func(rewards)
        return received_reward


class WeakStrongBanditAlgo(object):
    """
    Attributes:
        arm_nums: List[int]. The numbers of arms for all devices
        device_num: int. The number of devices
        pulls: Dict. Information about different types of pulls
        rounds: same as in Simulator
        budgets: same as in Simulator
        algo:

        T: current round
        B: current budgets 
        strong_pull_record: List[List[List[float]]]. records = strong_pull_record[device][arm]
        weak_pull_record: Dict[tuple: List[float]]. records = weak_pull_record[(d1_a,d2_a,...)]

    Methods:
        The most important method is next_pull(), which captures the whole essence of a MAB algorithm.
    """
    def __init__(self, arm_nums, pulls, rounds, budgets, algo=None):
        self.arm_nums = arm_nums
        self.device_num = len(arm_nums)
        self.pulls = pulls
        self.budgets = budgets
        self.algo = algo
        self.strong_pull_cost = pulls["strong"]
        self.weak_pull_cost = pulls["weak"]
        self.strong_rounds = self.budgets // self.strong_pull_cost
        self.rounds = rounds  # + self.strong_rounds
        self.initialize()
        self.algo_name = "WeakStrongBanditAlgo"

    def initialize(self):
        self.T = 0
        self.B = 0

        # initialize records
        # weak records initialization: Dict[(arms,): (mean, times)]
        def rec_loop(n, arms_list):
            if n >= 0:
                for i in range(self.arm_nums[n]):
                    new_list = [i] + arms_list
                    rec_loop(n-1, new_list)
            else:
                self.weak_pull_record[tuple(arms_list)] = (0, 0)
        self.weak_pull_record = {}
        rec_loop(self.device_num-1,[])

        # strong records initialization: List[List[(mean, times)]]
        self.strong_pull_record = []
        for i in range(self.device_num):
            r = [(0, 0) for x in range(self.arm_nums[i])]
            self.strong_pull_record.append(r)

    def choose_new_arm(self, response_of_last_round):
        self.update_info(response_of_last_round)
        return self.next_pull()

    # Important method!
    def next_pull(self):
        """
        Chooses an arm according to the records by the MAB algo. Returns (pull_type, needed_info)
        """
        # if self.algo is not None:
        #     return self.algo(self.weak_pull_record, self.strong_pull_record)
        # return ("strong", 0, 0)
        self.T += 1
        return self.next_weak_pull()

    def next_strong_pull(self):
        self.B +=1
        if self.B > self.strong_rounds:
            return False
        return "strong", 0, 0

    def next_weak_pull(self):
        if self.T > self.rounds:
            return False
        arms = tuple([0 for x in range(self.device_num)])
        return "weak", arms

    def update_info(self, response):
        """
        Args:
            response: (pull_type, *args).
                      if pull_type is "strong": ("strong", device, arm, reward)
                      if pull_type is "weak": ("weak, arms, reward)
        """
        if response is None:
            return   
        if response[0] == "strong":
            device, arm, reward = response[1:]
            self.update_strong_pull_record(device, arm, reward)
        if response[0] == "weak":
            arms, reward = response[1:]
            self.update_weak_pull_record(arms, reward)

    def update_strong_pull_record(self, device, arm, reward):
        mean, times = self.strong_pull_record[device][arm]
        mean = (mean * times + reward) / float(times + 1)
        self.strong_pull_record[device][arm] = (mean, times + 1)
    
    def update_weak_pull_record(self, arms, reward):
        records = self.weak_pull_record
        if arms not in records.keys():
            records[arms] = (0, 0)
        mean, times = records[arms]
        mean = (mean * times + reward) / float(times + 1)
        records[arms] = (mean, times + 1)

    def get_records_of_arms(self, arms):
        records = self.weak_pull_record
        if arms not in records.keys():
            records[arms] = (0, 0)
        return records[arms]
    
    def get_records_of_device_arm(self, device, arm):
        return self.strong_pull_record[device][arm]


def explore_experiment():
    def reward_generator(reward):
        st_dev = 0.1
        r = np.random.normal(reward, st_dev)
        if r > 1:
            r = 1
        if r < 0:
            r = 0
        return r

    devices = [[0.5, 1], [0.4, 1]]
    pulls = {"strong": 3, "weak": 1}
    rounds = 5
    budgets = 0
    simu = Simulator(devices, pulls, rounds, budgets, generation=reward_generator)
    simu.start()


if __name__ == "__main__":
    explore_experiment()