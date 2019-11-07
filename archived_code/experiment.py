from naive_algo import *


class SingleMABWithStrongPull(WeakStrongBanditAlgo):
    """
    Only considers one bandit. Every K rounds it can incur a strong pull freely.
    Weak pulls are in fact stored in a strong pull record, considering strong_pull_record is a list while weak's is dict
    """
    def __init__(self, arm_nums, pulls, rounds, budgets, multiplier=1, algo=None, k=None, complexity=None):
        super(SingleMABWithStrongPull, self).__init__(arm_nums, pulls, rounds, budgets, algo)
        if k is None or k <= 0:
            self.k = 0
            self.k_rounds = 0
        else:
            self.k = k
            self.k_rounds = rounds // k
        self.complexity = False if complexity is None else complexity
        self.algo_name = f"k={self.k}"
        self.T = 1
        self.is_strong_pulled = True
        self.multiplier = multiplier

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
        def compute_H(device, arm_info, multiplier=self.multiplier):
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            complexity = diff + math.sqrt(1 / (2 * times))
            return multiplier*multiplier / float(complexity * complexity)

        def compute_B(device, a, arm_info):
            # TODO: change policies
            mean, times = arm_info
            confidence = self.multiplier * math.sqrt(a / times)
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            best_arm_identification_score = -diff + confidence
            best_arm_identification_with_bad_arms = diff + confidence
            best_arm_identification_score_original = mean + confidence
            best_arm_identification_confidence = confidence
            best_arm_identification_confidence_minus_mean = -mean + best_arm_identification_confidence


            largest_confidence = self.multiplier * math.sqrt(2*math.log(self.T)/float(times))
            smallest_mean = -mean
            largest_confidence_minus_mean = -mean + largest_confidence
            smallest_confidence = mean - largest_confidence

            score = best_arm_identification_score_original
            confidence
            return score, confidence
        n = self.arm_num
        rounds = self.rounds + self.k_rounds
        # rounds = math.ceil(self.T * (1 + 1 / k))+2
        # initialization of pulls
        result = self.has_arm_unexplored()
        device = 0
        if result is 0 or result:
            arm = result
            snapshot_of_confidence = [0 for x in range(self.arm_nums[device])]
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
                scores = [compute_B(i, a, x)[0] for x in self.strong_pull_record[i]]
                best_score = max(scores)
                # index = scores.index(best_score)
                index = self.randomized_index_of_max(best_score, scores)
                candidates.append((best_score, index))
            snapshot_of_confidence = scores
            best = max(candidates, key=itemgetter(0))
            device = candidates.index(best)
            arm = best[1]
        return "strong", device, arm, snapshot_of_confidence

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
            confidence = self.multiplier * math.sqrt(2*math.log(self.T)/float(times))
            # TODO: change to see what is better
            explore_rate = confidence
            # explore_rate = confidence/float(mean+confidence)
            return mean + confidence, explore_rate

        n = self.arm_num
        rounds = self.rounds + self.k_rounds

        n = self.arm_num
        # initialization of pulls
        result = self.has_arm_unexplored()
        if result is 0 or result:
            arm = result
            # TODO: change correspondingly
            explore_rate = 0
            snapshot_for_condifence = [0 for x in range(n)]
        else:
            scores = [compute_UCB(0, x)[0] for x in range(n)]
            snapshot_for_condifence = [compute_UCB(0, a)[1] for a in range(n)]
            arm = self.randomized_index_of_max(max(scores), scores)
            # explore_rate = compute_UCB(0, arm)[1]
        return "weak", tuple([arm]), snapshot_for_condifence

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

    def randomized_index_of_max(self, max_value, values):
        max_indices = []
        for i in range(len(values)):
            if values[i] == max_value:
                max_indices.append(i)
        return np.random.choice(max_indices)

    def get_second_best_arm_index(self, scores):
        best_arm_index = self.randomized_index_of_max(max(scores), scores)
        scores[best_arm_index] = float("-inf")
        second_index = self.randomized_index_of_max(max(scores), scores)
        return second_index


class UCB(SingleMABWithStrongPull):
    def next_pull(self):
        self.T += 1
        return self.next_weak_pull()


class UCBExploration(SingleMABWithStrongPull):
    def next_pull(self):
        self.T += 1
        return self.next_weak_pull()

    def next_weak_pull(self):
        strong_type, device, arm, snapshot_of_confidence = self.next_strong_pull()
        return "weak", tuple([arm]), snapshot_of_confidence


class EpsilonGreedy(SingleMABWithStrongPull):
    epsilon = None
    description = "Normal: epsilon greedy. Free: pure explore."

    def __init__(self, arm_nums, pulls, rounds, budgets, algo=None, k=None, multiplier=1, complexity=None,
                 epsilon=None):
        super(EpsilonGreedy, self).__init__(arm_nums, pulls, rounds, budgets, algo=algo, k=k, multiplier=multiplier,
                                            complexity=complexity)
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

    def next_weak_pull(self):
        n = self.arm_num
        rounds = self.rounds + self.k_rounds

        # initialization of pulls
        result = self.has_arm_unexplored()
        snapshot_for_condifence = [0 for x in range(n)]
        if result is 0 or result:
            arm = result
        else:
            if np.random.rand() >= self.epsilon:
                scores = [x[0] for x in self.strong_pull_record[0]]
                arm = self.randomized_index_of_max(max(scores), scores)
            else:
                arm = self.random_pull()
        return "weak", tuple([arm]), snapshot_for_condifence

    def next_strong_pull(self):
        device = 0
        arm = self.random_pull()
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence

    def random_pull(self):
        return np.random.randint(0, self.arm_num)


class EpsilonGreedyWorst(EpsilonGreedy):
    description = "Normal: epsilon greedy. Free: always worst."

    def next_strong_pull(self):
        device = 0
        result = self.has_arm_unexplored()
        scores = [x[0] for x in self.strong_pull_record[0]]
        arm = self.randomized_index_of_max(min(scores), scores)
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


class EpsilonGreedyBest(EpsilonGreedy):
    description = "Normal: epsilon greedy. Free: always best."

    def next_strong_pull(self):
        device = 0
        scores = [x[0] for x in self.strong_pull_record[0]]
        arm = self.randomized_index_of_max(max(scores), scores)
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


class EpsilonGreedyRealWorst(EpsilonGreedy):
    description = "Normal: epsilon greedy. Free: always real worst, which is arm 0."

    def next_strong_pull(self):
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", 0, 0, snapshot_of_confidence


class EpsilonGreedyRealBest(EpsilonGreedy):
    description = "Normal: epsilon greedy. Free: always real best, which is arm 8."

    def next_strong_pull(self):
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", 0, 8, snapshot_of_confidence


class EpsilonGreedyRealSecondBest(EpsilonGreedy):
    # Must modify second_best_index!
    second_best_index = None
    description = f"Normal: epsilon greedy. Free: always real second best."

    def next_strong_pull(self):
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", 0, EpsilonGreedyRealSecondBest.second_best_index, snapshot_of_confidence


class EpsilonGreedyUCB(EpsilonGreedy):
    description = "Normal: epsilon greedy. Free: UCB."

    def next_strong_pull(self):
        def compute_UCB(device, arm):
            mean, times = self.strong_pull_record[device][arm]
            # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
            confidence = self.multiplier * math.sqrt(2*math.log(self.T)/float(times))
            # TODO: change to see what is better
            explore_rate = confidence
            # explore_rate = confidence/float(mean+confidence)
            return mean + confidence, explore_rate

        device = 0
        result = self.has_arm_unexplored()
        if result is 0 or result:
            arm = result
        else:
            scores = [compute_UCB(0, i)[0] for i in range(self.arm_num)]
            arm = self.randomized_index_of_max(max(scores), scores)
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


class EpsilonGreedyUCBSecond(EpsilonGreedy):
    description = "Normal: epsilon greedy. Free: UCB second best."

    def next_strong_pull(self):
        def compute_UCB(device, arm):
            mean, times = self.strong_pull_record[device][arm]
            # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
            confidence = self.multiplier * math.sqrt(2*math.log(self.T)/float(times))
            # TODO: change to see what is better
            explore_rate = confidence
            # explore_rate = confidence/float(mean+confidence)
            return mean + confidence, explore_rate

        device = 0
        result = self.has_arm_unexplored()
        if result is 0 or result:
            arm = result
        else:
            scores = [compute_UCB(0, i)[0] for i in range(self.arm_num)]
            arm = self.get_second_best_arm_index(scores)
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


class UpdateNoConfidence(SingleMABWithStrongPull):
    def update_strong_pull_record(self, device, arm, reward):
        arms = self.strong_pull_record[device]
        mean, times = arms[arm]
        mean = (mean * times + reward) / float(times + 1)
        if times == 0:
            times = 1
        self.strong_pull_record[device][arm] = (mean, times)
        best_arm = max(arms, key=itemgetter(0))
        arms_copy = arms[:]
        arms_copy.remove(best_arm)
        second_arm = max(arms_copy, key=itemgetter(0))
        self.strong_best_record[device] = best_arm[0]
        self.strong_second_record[device] = second_arm[0]

    def update_weak_pull_record(self, arms, reward):
        device = 0
        arm = arms[device]
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


class UCBPureExplore(SingleMABWithStrongPull):
    description = "Normal: UCB. Free: pure explore."

    def next_strong_pull(self):
        device = 0
        arm = self.random_pull()
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence

    def random_pull(self):
        return np.random.randint(0, self.arm_num)


class UCBWithWorst(UCBPureExplore):
    description = "Normal: UCB. Free: always worst."

    def next_strong_pull(self):
        device = 0
        scores = [x[0] for x in self.strong_pull_record[0]]
        arm = self.randomized_index_of_max(min(scores), scores)
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


class UCBWithBest(UCBPureExplore):
    description = "Normal: UCB. Free: always best."

    def next_strong_pull(self):
        device = 0
        scores = [x[0] for x in self.strong_pull_record[0]]
        arm = self.randomized_index_of_max(max(scores), scores)
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


class UCBWithSecondBest(UCBPureExplore):
    description = "Normal: UCB. Free: always second best."

    def next_strong_pull(self):
        device = 0
        # result = self.has_arm_unexplored()
        # if result is 0 or result:
        #     arm = result
        # else:
        scores = [x[0] for x in self.strong_pull_record[0]]
        arm = self.get_second_best_arm_index(scores)
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


class UCBWithRealWorst(UCBPureExplore):
    description = "Normal: UCB. Free: always real worst."

    def next_strong_pull(self):
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", 0, 0, snapshot_of_confidence


class UCBWithRealBest(UCBPureExplore):
    description = "Normal: UCB. Free: always real best."

    def next_strong_pull(self):
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", 0, 8, snapshot_of_confidence


class UCBWithRealSecondBest(UCBPureExplore):
    description = "Normal: UCB. Free: always real second best."

    def next_strong_pull(self):
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", 0, SECOND_BEST_INDEX, snapshot_of_confidence


class UCBWithUCBSecondBest(UCBPureExplore):
    description = "Normal: UCB. Free: UCB second best."

    def next_strong_pull(self):
        def compute_UCB(device, arm):
            mean, times = self.strong_pull_record[device][arm]
            # print("({0},{1}) times: {2}, mean: {3}".format(device, arm, times, mean))
            confidence = self.multiplier * math.sqrt(2*math.log(self.T)/float(times))
            # TODO: change to see what is better
            explore_rate = confidence
            # explore_rate = confidence/float(mean+confidence)
            return mean + confidence, explore_rate

        device = 0
        result = self.has_arm_unexplored()
        if result is 0 or result:
            arm = result
        else:
            scores = [compute_UCB(0, i)[0] for i in range(self.arm_num)]
            arm = self.get_second_best_arm_index(scores)
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


class UCBWithEpsilonGreedy(UCBPureExplore):
    description = "Normal: UCB. Free: epsilon greedy."
    epsilon = None

    def __init__(self, arm_nums, pulls, rounds, budgets, algo=None, k=None, multiplier=1, complexity=None,
                 epsilon=None):
        super(UCBWithEpsilonGreedy, self).__init__(arm_nums, pulls, rounds, budgets, algo=algo, k=k,
                                                   multiplier=multiplier, complexity=complexity)
        if epsilon is None:
            epsilon = UCBWithEpsilonGreedy.epsilon
            assert epsilon is not None
        self.epsilon = self.original_epsilon = epsilon
        # if k is None or k < 1:
        #     self.epsilon = self.original_epsilon
        # else:
        #     self.epsilon = (self.original_epsilon * (self.k + 1) - 1) / k
        #     # self.epsilon = (k * self.original_epsilon - 1) / k
        # print(f"adjusted eplision is {self.epsilon}")

    def next_strong_pull(self):
        n = self.arm_num
        rounds = self.rounds + self.k_rounds

        # initialization of pulls
        result = self.has_arm_unexplored()
        if result is 0 or result:
            arm = result
        else:
            if np.random.rand() >= self.epsilon:
                scores = [x[0] for x in self.strong_pull_record[0]]
                arm = self.randomized_index_of_max(max(scores), scores)
            else:
                arm = self.random_pull()
        device = 0
        snapshot_of_confidence = [0 for i in range(self.arm_num)]
        return "strong", device, arm, snapshot_of_confidence


def get_algo_classes_with_epsilon(algo_classes, epsilon):
    return [get_algo_class_with_epsilon(algo, epsilon) for algo in algo_classes]


def get_algo_class_with_epsilon(cls, epsilon):
    name = f"{cls.__name__}WithEpsilon{epsilon}"
    return type(name, (cls,), {"epsilon": epsilon})


def get_algo_class_with_second_best(cls, second_best_index):
    name = f"{cls.__name__}WithSecondBest{second_best_index}"
    return type(name, (cls,), {"second_best_index": second_best_index})


class SimulatorWithK(Simulator):
    def __init__(self, devices, pulls, rounds, budgets, multiplier=1, algos=None, generation=None, reward_func=None,
                 strong_pull_func=None, weak_pull_func=None, k=None, complexity=None):
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
            if r > multiplier:
                r = multiplier
            if r < 0:
                r = 0
            return r

        self.devices = MultiDevices(devices, reward_generator, reward_func)
        self.pulls = pulls
        self.rounds = rounds
        self.budgets = budgets
        self.arm_nums = self.devices.get_arm_nums()
        arm_nums = self.arm_nums
        self.algos = []
        if algos is None:
            self.algos = [WeakStrongBanditAlgo(arm_nums, pulls, rounds, budgets)]
        elif isinstance(algos, list):
            for algo in algos:
                self.algos.append(algo(arm_nums, pulls, rounds, budgets, multiplier=self.multiplier))
        elif issubclass(algos, SingleMABWithStrongPull):
            for i in range(1, k+1):
                self.algos.append(algos(arm_nums, pulls, rounds, budgets, multiplier=self.multiplier, k=i,
                                        complexity=complexity))
            self.algos.append(algos(arm_nums, pulls, rounds, budgets, multiplier=self.multiplier, k=0,
                                    complexity=complexity))
        else:
            self.algos = [algos(arm_nums, pulls, rounds, budgets, multiplier=self.multiplier)]
        self.strong_pull_func = strong_pull_func
        self.weak_pull_func = weak_pull_func

    def __run_algo__(self, algo, interval=None, silent=True, explore_rate_record=False):
        algo.initialize()
        total_reward = 0
        intervaled_regret = []
        intervaled_explore_rates = []
        response = None
        best_reward = self.devices.get_best_reward()
        explore_rates = [(0, 0) for i in range(self.arm_nums[0])] if explore_rate_record else None
        # explore_rate_detail_record = [[] for i in range(self.arm_nums[0])]
        if not silent:
            print("The best reward is {0}".format(best_reward))
        for i in range(1, self.rounds + 1):
            chosen_arm = algo.choose_new_arm(response)
            assert chosen_arm
            pull_type, *selection_info = chosen_arm
            while pull_type == "strong":
                device, arm, snapshot_of_strong_explore_rate = selection_info
                reward = self.devices.strong_pull(device, arm, self.strong_pull_func)
                response = (pull_type, device, arm, reward)
                if not silent:
                    print("round {0}, response is {1}".format(i, response))
                chosen_arm = algo.choose_new_arm(response)
                assert chosen_arm
                pull_type, *selection_info = chosen_arm
            if pull_type == "weak":
                arms = selection_info[0]
                arm = arms[0]
                if explore_rate_record:
                    snapshot_of_exploration_rate = selection_info[1]
                    explore_rate_of_chosen = snapshot_of_exploration_rate[arm]
                    old_info = explore_rates[arm]
                    explore_rates[arm] = self.update_exploratin_info(old_info, explore_rate_of_chosen)
                reward = self.devices.weak_pull(arms, self.weak_pull_func)
                response = (pull_type, arms, reward)
            else:
                raise TypeError("Pull_type should be 'strong' or 'weak'.")
            if not silent:
                print("round {0}, response is {1}".format(i, response))
            total_reward += reward
            # handle about intervals
            if interval is not None and i % interval == 0:
                intervaled_regret.append(best_reward * i - total_reward)
                if explore_rate_record:
                    rates_copy = explore_rates[:]
                    intervaled_explore_rates.append(rates_copy)
        if interval is None or self.rounds % interval != 0:
            regret = best_reward * self.rounds - total_reward
            intervaled_regret.append(regret)
            if explore_rate_record:
                intervaled_explore_rates.append(explore_rates)
        if not silent:
            print("after {0} rounds, the total regret is {1} for {2}".format(self.rounds, regret, algo.algo_name))
        return intervaled_regret, intervaled_explore_rates

    def update_exploratin_info(self, old_info, new_rate):
        mean, times = old_info
        mean = (mean*times + new_rate)/(times+1)
        return mean, times+1

    def run(self, times, interval=None, silent=True, explore_rate_record=False):
        if interval is None:
            interval = self.rounds
        all_intervaled_regrets = []
        algo_names = []
        explore_rates = []
        x_axis_length = int((self.rounds - 0.5) // interval) + 1
        for algo in self.algos:
            algo_intervaled_regrets = np.zeros(x_axis_length)
            if explore_rate_record:
                explore_rates_for_single_algo = np.zeros([x_axis_length, self.arm_nums[0]])
            for i in range(times):
                single_run_intervaled_regrets, explore_rate = self.__run_algo__(algo, interval, silent, explore_rate_record)
                algo_intervaled_regrets = algo_intervaled_regrets + np.array(single_run_intervaled_regrets)
                if explore_rate_record:
                    for j, rate_of_one_interval in enumerate(explore_rate):
                        explore_rate[j] = [x[1] for x in rate_of_one_interval]
                    explore_rate_without_confidence = np.array(explore_rate) # list of pulling times for ams
                    explore_rates_for_single_algo = explore_rates_for_single_algo + explore_rate_without_confidence
                # explore_details.append(explore_detail)
            average_intervaled_regrets = algo_intervaled_regrets / times
            all_intervaled_regrets.append(average_intervaled_regrets)
            algo_names.append(algo.algo_name)
            if explore_rate_record:
                explore_rates.append(explore_rates_for_single_algo / times)
        all_intervaled_regrets = np.array(all_intervaled_regrets).transpose()
        regrets = np.around(all_intervaled_regrets, 2).tolist()
        if explore_rate_record:
            explore_rates = np.round(np.array(explore_rates), 1).transpose(1, 0, 2).tolist()
        # start to output
        print(algo_names)
        for i in range(x_axis_length):
            print(f"after {min((i+1)*interval, self.rounds)} round")
            print("the average regrets are ", regrets[i])
            if explore_rate_record:
                print("the number of pulls for arms are :")
                for algo, record in zip(algo_names, explore_rates[i]):
                    print(f"{algo} pulls: {record}")
            print("")
        return algo_names, regrets, explore_rates


if __name__ == "__main__":
    def reward_generator(st_dev=0.2, multiplier=1):
        def generator(r):
            r = np.random.normal(r, st_dev*multiplier)
            if r > multiplier:
                r = multiplier
            if r < 0:
                r = 0
            return r
        return generator

    def compute_true_H(devices, multiplier=1):
        def compute_H_for_single_arm(arm, best, second, multiplier):
            diff = best - arm if arm - best != 0 else best - second
            # complexity = diff + math.sqrt(1 / (2 * times))
            complexity = diff
            return multiplier*multiplier / float(complexity * complexity)

        H = 0
        for device in devices:
            best = max(device)
            copy = device[:]
            copy.remove(best)
            second = max(copy)
            H += sum([compute_H_for_single_arm(a, best, second, multiplier) for a in device])
        return H

    def get_second_max(arr):
        arr_copy = arr[:]
        l = max(arr)
        arr_copy.remove(l)
        return arr.index(max(arr_copy))

    st_dev = 0.1
    multiplier = 10

    devices = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    devices = [[0.4, 0.5, 0.6]]
    devices = [[0.4, 0.5]]
    devices = [[multiplier * x for x in devices[0]]]
    pulls, budgets = {"strong": 1, "weak": 1}, 11
    complexity = compute_true_H(devices, multiplier=multiplier)
    # complexity = None

    SECOND_BEST_INDEX = get_second_max(devices[0])
    # EPSILON = 0.2
    rounds = 50
    k = 1
    trials = 1000
    interval = 1

    silent = True
    explore_rate_record = False

    print(f'{trials} trials. Each trial with {rounds} rounds.')

    epsilon_list = [0.01, 0.02, 0.05]
    epsilon_list = [0]

    # ploting
    x_axis_length = int((rounds - 0.5) // interval) + 1
    x_axis = [i*interval for i in range(1, x_axis_length)]
    x_axis.append(rounds)
    plt.figure()
    plt.xlabel("rounds")
    plt.ylabel("Regrets diff between maximum-free-pull and no-free-pull")
    plt.title(f"{trials} trials. {len(devices[0])} arms")
    for e in epsilon_list:
        EpsilonGreedy.epsilon = e
        print(f"the epsilon is {EpsilonGreedyRealBest.epsilon}")
        EpsilonGreedyRealSecondBest.second_best_index = SECOND_BEST_INDEX
        print(f"the second best arm index is {EpsilonGreedyRealSecondBest.second_best_index}")
        experiment_algos = [EpsilonGreedyRealSecondBest]
        for algos in experiment_algos:
            print(algos.description)
            simu = SimulatorWithK(devices, pulls, rounds, budgets, multiplier=multiplier, algos=algos,
                                  generation=reward_generator(st_dev=st_dev, multiplier=multiplier), k=k,
                                  complexity=complexity)
            names, regrets, explore_rates = simu.run(trials, interval=interval, silent=silent,
                                                     explore_rate_record=explore_rate_record)
            regrets_diff = [regret[0]-regret[-1] for regret in regrets]
            plt.plot(x_axis, regrets_diff, label=f"{algos.__name__}. epsilon {e}")
    plt.legend(loc="upper left")
    plt.savefig("./experiment2.pdf")
    plt.show()

    # # UCB-Exploration
    # algos = UpdateNoConfidence
    # simu = SimulatorWithK(devices, pulls, rounds, budgets, multiplier=multiplier, algos=algos,
    #                       generation=reward_generator(st_dev=st_dev, multiplier=multiplier), k=k,
    #                       complexity=complexity)
    # names, regrets, explore_details, explore_rates = simu.run(trials, silent=True, explore_rate_record=False)








    # arms_num = len(devices[0])
    # x = [i for i in range(1, rounds+1)]
    # plots = []
    # for i in range(k+1):
    #     plt.figure()
    #     rates = explore_details[i]
    #     for j in range(arms_num):
    #         plt.plot(x, rates[j], label="arm {0}".format(j+1))
    #     plt.title(names[i])
    #     plt.legend(loc='lower left')
    #     plt.show()

