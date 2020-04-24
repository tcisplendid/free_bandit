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
def plot(xlabel, ylabel, title, x, y_dict, save_path=None):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    for label, y in y_dict.items():
        plt.plot(x, y, label=label)
    plt.legend(loc="best")
    if save_path:
        plt.savefig(f"{save_path}/{title}.png")
    plt.show()

class Experiment(object):
    def __init__(self, experiment_options, log_options):
        self.experiment_options = experiment_options
        self.log_options = log_options
        rounds = self.experiment_options["rounds"]
        k = self.experiment_options["k"]
        interval = self.experiment_options["interval"]

        # ploting x-axis
        x_axis_length = int((rounds - 0.5) // interval) + 1
        self.x_axis = [i * interval for i in range(1, x_axis_length)] + [rounds]

    def compare_free_policies(self, device, base_algo, free_policies, plot_options):
        k = self.experiment_options["k"]
        xlabel = "Rounds"
        ylabel = "Reduced regrets by free pull"
        title = plot_options['title']
        issave = plot_options["save_data"]
        saved_path = self.create_saving_directory(title)

        y_dict = {}

        # experiment: base algo with all policies
        simu = Simulator(device, self.experiment_options)

        # ploting
        default_regrets, _ = simu.run(base_algo, free_policy=None, log_options=self.log_options)
        y_dict["pure explore"] = default_regrets
        for policy_name, policy in free_policies.items():
            regrets, pulling_times = simu.run(base_algo, free_policy=policy, log_options=self.log_options)
            if issave:
                self.save_json(saved_path, device, base_algo.__name__, policy_name, (regrets, pulling_times),
                               self.experiment_options, self.log_options)
            y_dict[policy_name] = regrets
        if not issave: saved_path = None
        plot(xlabel, ylabel, title, self.x_axis, y_dict, saved_path)

    def compare_k(self, device, base_algo, free_policy, plot_options):
        k = self.experiment_options["k"]
        xlabel = "Rounds"
        ylabel = "Accumulative regrets"
        title = plot_options['title']
        issave = plot_options["save_data"]
        saved_path = self.create_saving_directory(title)

        y_dict = {}
        simu = Simulator(device, self.experiment_options)

        all_regrets, all_pulling_times = simu.run(base_algo, free_policy=free_policy, log_options=self.log_options)
        # TODO: to delete
        if issave:
            self.save_json(saved_path, device, base_algo.__name__, free_policy.__name__, (all_regrets, all_pulling_times),
                           self.experiment_options, self.log_options)
        for i, regrets in enumerate(all_regrets):
            if i == len(all_regrets) - 1:
                y_dict["no free pull"] = regrets
            else:
                y_dict[f"k={i+1}"] = regrets
        if not issave: saved_path = None
        plot(xlabel, ylabel, title, self.x_axis, y_dict, saved_path)

    def compare_base_algos(self, device, base_algos, free_policy, plot_options, k=None):
        if k is None:
            k = self.experiment_options["k"]
        tmpk = self.experiment_options["k"]
        self.experiment_options["k"] = k
        xlabel = "Rounds"
        ylabel = "Accumulative regrets"
        title = plot_options['title']
        issave = plot_options["save_data"]
        saved_path = self.create_saving_directory(title)

        # experiment: base algo with all policies
        simu = Simulator(device, self.experiment_options)
        y_dict = {}

        for base_algo_name, base_algo in base_algos.items():
            tmp = None
            if isinstance(base_algo, tuple):
                tmp = base_algo[0].epsilon
                base_algo[0].epsilon, base_algo = base_algo[1], base_algo[0]
            regrets, pulling_times = simu.run(base_algo, free_policy=free_policy, log_options=self.log_options)
            if issave:
                self.save_json(saved_path, device, base_algo.__name__, free_policy.__name__, (regrets, pulling_times),
                               self.experiment_options, self.log_options)
            y_dict[base_algo_name] = regrets[k-1]
            if tmp is not None:
                base_algo.epsilon, tmp = tmp, None
        self.experiment_options["k"] = tmpk
        if not issave: saved_path = None
        plot(xlabel, ylabel, title, self.x_axis, y_dict, saved_path)


    def plot(self, device, base_algo, free_policies, plot_options):
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

    def plot_base_algo(self, device, base_algos, plot_options, free_policy=None, k=0):
        rounds = self.experiment_options["rounds"]
        tmpk = self.experiment_options["k"]
        self.experiment_options["k"] = k
        # trials = self.experiment_options["trials"]
        interval = self.experiment_options["interval"]
        title = plot_options['title']
        saved_path = self.create_saving_directory(title)

        # experiment: base algo with all policies
        simu = Simulator(device, self.experiment_options)

        # ploting
        x_axis_length = int((rounds - 0.5) // interval) + 1
        x_axis = [i * interval for i in range(1, x_axis_length)]
        x_axis.append(rounds)

        plt.figure()
        plt.xlabel("Rounds")
        plt.ylabel("Accumulative regrets")
        plt.title(f"{plot_options['title']}")
        for base_algo_name, base_algo in base_algos.items():
            if isinstance(base_algo, tuple):
                base_algo[0].epsilon, base_algo = base_algo[1], base_algo[0]
            regrets, pulling_times = simu.run(base_algo, free_policy=free_policy, log_options=self.log_options)
            if self.log_options["persistence"]:
                self.save_json(saved_path, base_algo.__name__, str(free_policy), (regrets, pulling_times),
                               self.experiment_options, self.log_options, plot_options)
            plt.plot(x_axis, regrets[k-1], label=f"{base_algo_name}")
        plt.legend(loc="best")
        if plot_options["save_png"]:
            plt.savefig(f"{saved_path}/{plot_options['title']}.png")
        plt.show()
        self.experiment_options["k"] = tmpk



    @staticmethod
    def create_saving_directory(title):
        today = date.today().strftime("%m-%d")
        path = f"./results/{today}-{title}"
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        finally:
            return path

    @staticmethod
    def save_json(path, device, base_algo, policy_name, results, experiment_options, log_options):
        tosave = {
            "arms": device.arms,
            "algo": (base_algo, policy_name),
            "results": results,
            "settings": {
                "experiment_options": experiment_options,
                "log_options": log_options,
            }
        }
        path = f"{path}/{base_algo}-{policy_name}.json"
        with open(path, 'w') as f:
            json.dump(tosave, f, ensure_ascii=False, indent=4)


def plot_reward_distribution(reward_settings):
    x_axis = list(range(60, 77))

    plt.figure()
    plt.xlabel("Temperature")
    plt.ylabel("Expected average satisfaction level")
    plt.title(f"Reward Distribution Settings")
    for name, rewards in reward_settings.items():
        print(rewards)
        plt.plot(x_axis, rewards, label=f"{name}")
    # plt.legend(loc="best")
    plt.legend(loc="lower left")
    plt.show()


def plot_counterexample():
    arms = [0.1, 0]
    st_dev = -1
    SimpleDevice.variance = 1
    device = SimpleDevice(arms)
    experiment_options = {
        "rounds": 500,
        "k": 1,
        "trials": 3000,
        "interval": 2,
        "st_dev": st_dev
    }

    log_options = {
        "silent": True,
        "log_pulling_times": False,
        "return_diff": False,  # True to compare diff policies
        "avg_regret": False,
        "persistence": False
    }

    plot_options = {
        "title": f"Simple Counter Example (Always Bad Arm)",
        "save_data": False
    }
    experiment = Experiment(experiment_options, log_options)
    EpsilonGreedy.epsilon = 0.05
    experiment.compare_k(device, EpsilonGreedy, real_second_best_generator(arms), plot_options)


def draw_report(arms, title, base):
    st_dev = -1
    device = Device(arms, st_dev)
    # device = TempScheduler(arms=arms, n=10)
    # print("arms: ", arms)

    experiment_options = {
        "rounds": 500,
        "k": 1,
        "trials": 1000,
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
        "title": f"{title}",  # "Two_peak_reward_distribution",
        "save_data": True
    }

    epsilon = 0.1
    EpsilonGreedy.epsilon = epsilon
    # UpperConfidenceBound.epsilon = epsilon
    # ThompsonSampling.epsilon = epsilon
    epsilon2 = 0.4

    # policies = policies_generator(arms)
    if base == -1:
        bases = {
            f"epsilon_Greedy({epsilon})": EpsilonGreedy,
            "Upper Confidence Bound": UpperConfidenceBound,
            "Thompson Sampling": ThompsonSampling,
            # f"epsilon_Greedy({epsilon2})": (EpsilonGreedy, epsilon2),
            # "random": RandomPull,
        }
        experiment_options["rounds"] = 1000
        log_options["return_diff"] = False
        experiment_options["trials"] = 200
        plot_options["title"] = f"No Free Pull ({title})"
        experiment = Experiment(experiment_options, log_options)
        experiment.compare_base_algos(device, bases, choose_nothing, plot_options, k=0)
        return


    all_policies = {
        # "same": ts_best,
        # "second": ts_second_best,
        # "worst": ts_worst,
        # "least pull": least_pulled,
        f"EG({epsilon})": epsilongreedy_policy_generator(epsilon),
        "UCB": ucb_best,
        "TS": ts_best,
        f"EG({epsilon2})": epsilongreedy_policy_generator(epsilon2),
    }


    if base == 1:
        # do eg simulation
        eg_policies = {
            # "same": epsilongreedy_policy_generator(epsilon),
            # "second": second_best_mean,
            # "worst(SE)": worst_mean_with_successive_elimination,
            # "least pull(SE)": least_pulled_with_successive_elimination,
            "UCB": ucb_best,
            "TS": ts_best,
            f"EG({epsilon2})": epsilongreedy_policy_generator(epsilon2),
        }
        policies = {"EG_policies": eg_policies,}
        plot_options["title"] = f"EG Normal Pulls ({title})"
        experiment_options["trials"] = 25000
        experiment = Experiment(experiment_options, log_options)
        experiment.compare_free_policies(device, EpsilonGreedy, policies, plot_options)
        # p.apply_async(proxy, args=(experiment, device, EpsilonGreedy, policies, plot_options))
    elif base == 2:
        # do ucb simulation
        ucb_policies = {
            # "same": ucb_best,
            # "second": ucb_second_best,
            # "worst(SE)": worst_ucb_with_successive_elimination,
            # "least pull(SE)": least_pulled_with_successive_elimination,
            f"EG({epsilon})": epsilongreedy_policy_generator(epsilon),
            "TS": ts_best,
            f"EG({epsilon2})": epsilongreedy_policy_generator(epsilon2),
        }
        policies = {"UCB_policies": ucb_policies,}
        plot_options["title"] = f"UCB Normal Pulls ({title})"
        experiment_options["trials"] = 15000
        experiment = Experiment(experiment_options, log_options)
        experiment.compare_free_policies(device, UpperConfidenceBound, policies, plot_options)
    # p.apply_async(proxy, args=(experiment, device, UpperConfidenceBound, policies, plot_options))

    elif base == 3:
        # do ts simulation
        ts_policies = {
            # "same": ts_best,
            # "second": ts_second_best,
            # "worst": ts_worst,
            # "least pull": least_pulled,
            f"EG({epsilon})": epsilongreedy_policy_generator(epsilon),
            "UCB": ucb_best,
            f"EG({epsilon2})": epsilongreedy_policy_generator(epsilon2),
        }

        policies = {"TS_policies": ts_policies,}
        plot_options["title"] = f"TS Normal Pulls ({title})"
        experiment_options["trials"] = 20000
        experiment = Experiment(experiment_options, log_options)
        experiment.compare_free_policies(device, ThompsonSampling, policies, plot_options)
    elif base == 4:
        policies = {"random_policies": all_policies, }
        plot_options["title"] = f"Random Normal Pulls ({title})"
        experiment_options["trials"] = 200 * 2
        experiment = Experiment(experiment_options, log_options)
        experiment.compare_free_policies(device, RandomPull, policies, plot_options)
    elif base == 5:
        eg2_policies = {
            # "same": epsilongreedy_policy_generator(epsilon),
            # "second": second_best_mean,
            # "worst(SE)": worst_mean_with_successive_elimination,
            # "least pull(SE)": least_pulled_with_successive_elimination,
            "UCB": ucb_best,
            "TS": ts_best,
            f"EG({epsilon})": epsilongreedy_policy_generator(epsilon),
        }
        policies = {"EG_policies": eg2_policies, }
        EpsilonGreedy.epsilon = epsilon2
        plot_options["title"] = f"EG({epsilon2}) Normal Pulls ({title})"
        experiment_options["trials"] = 25000
        experiment = Experiment(experiment_options, log_options)
        experiment.compare_free_policies(device, EpsilonGreedy, policies, plot_options)

    else:
        raise ValueError("wrong id")
        # do eg simulation
        # policies = {"EG_policies": eg_policies,}
        # experiment_options["trials"] = 25000
        # experiment = Experiment(experiment_options, log_options)
        # # experiment.plot(device, EpsilonGreedy, policies, plot_options)
        # p.apply_async(proxy, args=(experiment, device, EpsilonGreedy, policies, plot_options))
        # # do ucb simulation
        # policies = {"UCB_policies": ucb_policies,}
        # plot_options["title"] = f"UCB Normal Pulls {title}"
        # experiment_options["trials"] = 15000
        # experiment = Experiment(experiment_options, log_options)
        # experiment.plot(device, UpperConfidenceBound, policies, plot_options)
        # # p.apply_async(proxy, args=(experiment, device, UpperConfidenceBound, policies, plot_options))
        # # do ts simulation
        # policies = {"TS_policies": ts_policies,}
        # plot_options["title"] = f"TS Normal Pulls {title}"
        # experiment_options["trials"] = 20000
        # experiment = Experiment(experiment_options, log_options)
        # experiment.plot(device, ThompsonSampling, policies, plot_options)
    # p.apply_async(proxy, args=(experiment, device, ThompsonSampling, policies, plot_options))

def compare_reward_distribution():
    # rewards are calculated by weighted average of individual rewards.
    # different distributions for each people/group:
    # Gaussian distribution
    # piecewise distribution
    # sampled from two distributions (normal people, weak people)
    # sum of a kind of variables
    accumulative_absolute = [0.46918875, 0.51789875, 0.56212875, 0.60424375, 0.63678, 0.6647075, 0.684915, 0.69845875,
                             0.70131, 0.69772875, 0.684095, 0.664595, 0.63828625, 0.60396875, 0.56320625, 0.51690625,
                             0.46640875]

    # rewards are assigned semiarbitrarily to show some extreme cases
    one_peak = beta_shape_reward_generator(17, 100, 100, 0.2, 1)
    similar_normal = beta_shape_reward_generator(17, 2, 2, 0.7, 1)
    right_peak = beta_shape_reward_generator(17, 8, 2, 0.2, 0.4)
    two_peak = [x + y for x, y in
                zip(beta_shape_reward_generator(17, 10, 2, 0.1, 0.8), beta_shape_reward_generator(17, 2, 10, 0.1, 0.8))]

    reward_settings = {
        "basic": accumulative_absolute,
        # "one peak": one_peak,
        # "flat": similar_normal,
        # "skew peak": right_peak,
        # "two peaks": two_peak,
        # "(test)": [0.023, 0.03, 0.029, 0.001, 0.05, 0.06, 0.0234, 0.035, 0.01, 0.11]
    }

    p = Pool(10)
    for t, arr in reward_settings.items():
        bases = [1, 2, 3]
        bases = [5]
        bases = [-1]
        for i in bases:
            draw_report(arr, t, i)
            # p.apply_async(draw_report, args=(arr, t, i))
        # p.apply_async(draw_report, args=(arr, t, -1))
    p.close()
    p.join()

if __name__ == "__main__":
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
    # plot_counterexample()
    compare_reward_distribution()