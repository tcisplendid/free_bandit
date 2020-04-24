from strategies import *
import matplotlib.pyplot as plt
import json
from datetime import date
import os
from application import *
from multiprocessing import Pool, Process
import copy
from scipy.stats import beta

# cls = EpsilonGreedy
# name = f"{cls.__name__}WithWorstMean"
# EpsilonGreedyWorst = type(name, (cls,), {"free_pull": worst_mean})

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
            # if r > 1:
            #     r = 1
            # if r < 0:
            #     r = 0
            return r
        print("constructing device")
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

class Experiment(object):
    def __init__(self, experiment_options, log_options):
        print("constructing")
        self.experiment_options = experiment_options
        self.log_options = log_options

    def plot(self, device, base_algo, free_policies, plot_options):
        print("ploting")
        return
        rounds = self.experiment_options["rounds"]
        k = self.experiment_options["k"]
        trials = self.experiment_options["trials"]
        interval = self.experiment_options["interval"]
        title = plot_options['title']
        saved_path = self.create_saving_directory(title)

        # experiment: base algo with all policies
        simu = Simulator(device, self.experiment_options)

        # ploting
        x_axis_length = int((rounds - 0.5) // interval) + 1
        x_axis = [i * interval for i in range(1, x_axis_length)]
        x_axis.append(rounds)

        # set plotting different behavior
        if self.log_options["return_diff"]:
            default_regrets, _ = simu.run(base_algo, free_policy=None, log_options=self.log_options)

            # plot regret_diff among different free policies.
            # each plot compares a group of free policies with the same base algorithm
            for policies_group_name, policies_group in free_policies.items():
                plt.figure()
                plt.xlabel("Rounds")
                plt.ylabel("Reduced regrets by free pull")
                plt.title(f"{plot_options['title']}")

                plt.plot(x_axis, default_regrets, label=f"pure explore")
                for policy_name, policy in policies_group.items():
                    regrets, pulling_times = simu.run(base_algo, free_policy=policy, log_options=self.log_options)
                    if self.log_options["persistence"]:
                        self.save_json(saved_path, base_algo.__name__, policy_name, (regrets, pulling_times),
                                       self.experiment_options, self.log_options, plot_options)
                    plt.plot(x_axis, regrets, label=f"{policy_name}")
                plt.legend(loc="best")
                if plot_options["save_png"]:
                    plt.savefig(f"{saved_path}/{policies_group_name}.png")
                plt.show()
        else:
            # plot real regrets among k
            # each plot is the same base algo and free policy with different k
            for policies_group_name, policies_group in free_policies.items():
                for policy_name, policy in policies_group.items():
                    plt.figure()
                    plt.xlabel("Rounds")
                    plt.ylabel("Accumulative regrets")
                    plt.title(f"{plot_options['title']}")
                    all_regrets, all_pulling_times = simu.run(base_algo, free_policy=policy, log_options=self.log_options)
                    # TODO: to delete
                    if self.log_options["persistence"]:
                        self.save_json(saved_path, base_algo.__name__, policy_name, (all_regrets, all_pulling_times),
                                       self.experiment_options, self.log_options, plot_options)
                    for i, regrets in enumerate(all_regrets):
                        if i == len(all_regrets)-1:
                            i = "no free pull"
                        else:
                            i+=1
                        plt.plot(x_axis, regrets, label=f"k={i}")
                        # TODO: to delete
                        print(f"regrets k={i}: {regrets}")
                    plt.legend(loc="best")
                    if plot_options["save_png"]:
                        plt.savefig(f"{saved_path}/{policies_group_name}.png")
                    plt.show()


    @staticmethod
    def create_saving_directory(title):
        today = date.today().strftime("%m-%d")
        path = f"./results/test-{today}-{title}"
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        finally:
            return path

    @staticmethod
    def save_json(path, base_algo, policy_name, results, experiment_options, log_options, plot_options):
        tosave = {
            "algo": (base_algo, policy_name),
            "results": results,
            "settings": {
                "experiment_options": experiment_options,
                "log_options": log_options,
                "plot_options": plot_options
            }
        }
        path = f"{path}/{base_algo}-{policy_name}.json"
        with open(path, 'w') as f:
            json.dump(tosave, f, ensure_ascii=False, indent=4)


def proxy(e, *args):
    print("proxy")
    print(*args)
    e.plot(*args)


def draw_report(arms, title, p, base):

    st_dev = -1
    device = Device(arms, st_dev)
    # device = TempScheduler(arms=arms, n=10)
    # print("arms: ", arms)

    experiment_options = {
        "rounds": 500,
        "k": 1,
        "trials": 15000,
        "interval": 2,
        "st_dev": st_dev
    }

    log_options = {
        "silent": True,
        "log_pulling_times": False,
        "return_diff": True,  # True to compare diff policies
        "avg_regret": False,
        "persistence": False
    }

    plot_options = {
        "title": f"EG Normal Pulls {title}",  # "Two_peak_reward_distribution",
        "x_label": "rounds",
        "save_png": True
    }

    # device_options = {
    #     "device": TempScheduler,
    #     "args": {
    #         "n": 100
    #     }
    # }

    epsilon = 0.1
    EpsilonGreedy.epsilon = epsilon
    WorstEpsilonGreedy.epsilon = epsilon
    UpperConfidenceBound.epsilon = epsilon
    ThompsonSampling.epsilon = epsilon

    # policies = policies_generator(arms)

    eg_policies = {
        "same": epsilongreedy,
        "second": second_best_mean,
        "worst(SE)": worst_mean_with_successive_elimination,
        "least pull(SE)": least_pulled_with_successive_elimination,
        "UCB": ucb_best,
        "TS": ts_best
    }

    ucb_policies = {
        "same": ucb_best,
        "second": ucb_second_best,
        "worst(SE)": worst_ucb_with_successive_elimination,
        "least pull(SE)": least_pulled_with_successive_elimination,
        "EG(0.1)": epsilongreedy,
        "TS": ts_best
    }

    ts_policies = {
        "same": ts_best,
        "second": ts_second_best,
        "worst": ts_worst,
        "least pull": least_pulled,
        "EG(0.1)": epsilongreedy,
        "UCB": ucb_best
    }

    if base:
        # raise ValueError("wrong id")
        # do eg simulation
        policies = {"EG_policies": eg_policies,}
        experiment = Experiment(experiment_options, log_options)
        # experiment.plot(device, EpsilonGreedy, policies, plot_options)
        # p.apply_async(proxy, args=(experiment, device, EpsilonGreedy, policies, plot_options))
        # print(device)
        x = p.apply_async(proxy, args=(experiment, device, EpsilonGreedy, policies, plot_options))
        x.get()
        # do ucb simulation
        # policies = {"UCB_policies": ucb_policies,}
        # plot_options["title"] = f"UCB Normal Pulls {title}"
        # experiment_options["trials"] = 5000
        # experiment = Experiment(experiment_options, log_options)
        # # experiment.plot(device, UpperConfidenceBound, policies, plot_options)
        # # p.apply_async(proxy, args=(experiment, device, UpperConfidenceBound, policies, plot_options))
        # p.apply_async(experiment.plot, args=(device, UpperConfidenceBound, policies, plot_options))
        #
        # # do ts simulation
        # policies = {"TS_policies": ts_policies,}
        # plot_options["title"] = f"TS Normal Pulls {title}"
        # experiment_options["trials"] = 5000
        # experiment = Experiment(experiment_options, log_options)
        # # experiment.plot(device, ThompsonSampling, policies, plot_options)
        # p.apply_async(proxy, args=(experiment, device, ThompsonSampling, policies, plot_options))


if __name__ == "__main__":
    # arms = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9]
    # arms = [0.3, 0.4, 0.5]
    # arms = [0.5, 0.4]

    # rewards are calculated by weighted average of individual rewards.
    # different distributions for each people/group:
    # Gaussian distribution
    # piecewise distribution
    # sampled from two distributions (normal people, weak people)
    # sum of a kind of variables
    accumulative_absolute = [0.46918875, 0.51789875, 0.56212875, 0.60424375, 0.63678, 0.6647075, 0.684915, 0.69845875,
            0.70131, 0.69772875, 0.684095, 0.664595, 0.63828625, 0.60396875, 0.56320625, 0.51690625, 0.46640875]

    # rewards are assigned semiarbitrarily to show some extreme cases
    one_peak = beta_shape_reward_generator(17, 100, 100, 0.2, 1)
    similar_normal = beta_shape_reward_generator(17, 2, 2, 0.7, 1)
    right_peak = beta_shape_reward_generator(17, 8, 2, 0.2, 0.4)
    two_peak = [x+y for x, y in zip(beta_shape_reward_generator(17, 10, 2, 0.1, 0.8),beta_shape_reward_generator(17, 2, 10, 0.1, 0.8))]

    # simple dist
    random_best = [0, 0.1]
    random_second = [0.1, 0]
    arms2 = [0.5, 0.4]
    arms3 = [0.3, 0.4, 0.5]
    arms13 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9]

    # arms = arms13
    #
    # st_dev = 0.1
    # device = Device(arms, st_dev)
    # device = TempScheduler(arms=arms, n=10)
    reward_settings = {
        "(basic)": accumulative_absolute,
        "(one peak)": one_peak,
        "(flat)": similar_normal,
        "(skew peak)": right_peak,
        "(two peaks)": two_peak
    }
    # draw_report(arms13, "(test)")
    p = Pool()
    for t, arr in reward_settings.items():
        # for i in range(1, 4):
        draw_report(arr, t, p, True)
        #     p.apply_async(draw_report, args=(arr, t, i))
    p.close()
    p.join()



    # experiment_options = {
    #     "rounds": 100,
    #     "k": 1,
    #     "trials": 5000,
    #     "interval": 1,
    #     "st_dev": st_dev
    # }
    #
    # log_options = {
    #     "silent": True,
    #     "log_pulling_times": False,
    #     "return_diff": True,  # True to compare diff policies
    #     "avg_regret": False,
    #     "persistence": False
    # }
    #
    # plot_options = {
    #     "title": "Simulation of Epsilon Greedy Normal Pulls",  # "Two_peak_reward_distribution",
    #     "x_label": "rounds",
    #     "save_png": True
    # }

    # device_options = {
    #     "device": TempScheduler,
    #     "args": {
    #         "n": 100
    #     }
    # }
