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
        self.algo_name = "MAB with k = {0}".format(self.k)
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

        # TODO: just test! please delete later
        def compute_H(device, arm_info, multiplier=self.multiplier):
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            complexity = diff + math.sqrt(1 / (2 * times))
            return multiplier*multiplier / float(complexity * complexity)
        def compute_B(device, a, arm_info):
            # TODO: change policies
            mean, times = arm_info
            best_score = self.strong_best_record[device]
            diff = best_score - mean if mean - best_score != 0 else best_score - self.strong_second_record[device]
            best_arm_identification_score = -diff + self.multiplier * math.sqrt(a / times)
            best_arm_identification_with_bad_arms = diff + self.multiplier * math.sqrt(a / times)
            best_arm_identification_score_original = mean + self.multiplier * math.sqrt(a / times)

            largest_confidence = self.multiplier * math.sqrt(2*math.log(self.T)/float(times))
            smallest_mean = -mean
            largest_confidence_minus_mean = -mean + largest_confidence
            smallest_confidence = mean - largest_confidence

            score = best_arm_identification_score_original
            return score, self.multiplier * math.sqrt(a / times)
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
            if self.complexity:
                H = self.complexity
            else:
                H = 0
                for i in range(self.device_num):
                    H += sum([compute_H(i, a, self.multiplier) for a in self.strong_pull_record[i]])
            a = (25 / 36) * (rounds - n) / H
            scores = [compute_B(0, a, x)[0] for x in self.strong_pull_record[0]]
            snapshot_for_condifence = [compute_B(0, a, x)[1] for x in self.strong_pull_record[0]]
            scores = [compute_UCB(0, x)[0] for x in range(n)]
            snapshot_for_condifence = [compute_UCB(0, a)[1] for a in range(n)]
            arm = self.randomized_index_of_max(max(scores), scores)
            # explore_rate = compute_UCB(0, arm)[1]
        return "weak", tuple([arm]), snapshot_for_condifence