from strategies import *
import matplotlib.pyplot as plt
import json
from datetime import date
import os

# cls = EpsilonGreedy
# name = f"{cls.__name__}WithWorstMean"
# EpsilonGreedyWorst = type(name, (cls,), {"free_pull": worst_mean})


class Experiment(object):
    def __init__(self, experiment_options, log_options):
        self.experiment_options = experiment_options
        self.log_options = log_options

    def plot(self, arms, base_algo, free_policies, plot_options):
        rounds = self.experiment_options["rounds"]
        k = self.experiment_options["k"]
        trials = self.experiment_options["trials"]
        interval = self.experiment_options["interval"]
        title = plot_options['title']
        saved_path = self.create_saving_directory(title)

        # experiment: base algo with all policies
        simu = Simulator(arms, self.experiment_options)
        default_regrets, _ = simu.run(base_algo, free_policy=None, log_options=self.log_options)

        # ploting
        x_axis_length = int((rounds - 0.5) // interval) + 1
        x_axis = [i * interval for i in range(1, x_axis_length)]
        x_axis.append(rounds)

        # set plotting different behavior
        if log_options["return_diff"]:
            for policies_group_name, policies_group in free_policies.items():
                plt.figure()
                plt.xlabel("rounds")
                plt.ylabel("Changed regrets by max free pull")
                plt.title(f"{plot_options['title']}. {trials} trials.{len(arms)} arms")

                plt.plot(x_axis, default_regrets, label=f"pure_explore")
                for policy_name, policy in policies_group.items():
                    regrets, pulling_times = simu.run(base_algo, free_policy=policy, log_options=log_options)
                    if log_options["persistence"]:
                        self.save_json(saved_path, base_algo.__name__, policy_name, (regrets, pulling_times),
                                       experiment_options, log_options, plot_options)
                    plt.plot(x_axis, regrets, label=f"{policy_name}")
                plt.legend(loc="best")
                if plot_options["save_pdf"]:
                    plt.savefig(f"{saved_path}/{policies_group_name}.pdf")
                plt.show()

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


if __name__ == "__main__":
    arms = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9]
    arms = [0.3, 0.4, 0.5]

    experiment_options = {
        "rounds": 5000,
        "k": 1,
        "trials": 5000,
        "interval": 5
    }

    log_options = {
        "silent": True,
        "log_pulling_times": False,
        "return_diff": True,
        "avg_regret": False,
        "persistence": True
    }

    plot_options = {
        "title": "UCB_test",
        "x_label": "rounds",
        "save_pdf": True
    }

    EpsilonGreedy.epsilon = 0.02

    # policies = {
    #     "real_worst": real_worst_generator(arms),
    #     "ts_best": ts_best,
    #     "real_second_best": real_second_best_generator(arms)
    # }
    # policies = {"test_policies": policies}
    policies = policies_generator(arms)

    experiment = Experiment(experiment_options, log_options)
    experiment.plot(arms, UpperConfidenceBound, policies, plot_options)
