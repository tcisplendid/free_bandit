from strategies import *
import matplotlib.pyplot as plt

# cls = EpsilonGreedy
# name = f"{cls.__name__}WithWorstMean"
# EpsilonGreedyWorst = type(name, (cls,), {"free_pull": worst_mean})

if __name__ == "__main__":
    arms = [0.3, 0.4, 0.5, 0.6, 0.7]

    rounds = 2500
    k = 1
    trials = 10000
    interval = 5

    silent = True
    log_pulling_times = False

    base_algo = ThompsonSampling
    # Thompson Sampling bad cases test
    simu = Simulator(arms, rounds, 1)
    # default_regrets_diff, _ = simu.run(base_algo, free_policy=None, interval=interval, times=trials,
    #                                    silent=silent, log_pulling_times=log_pulling_times)

    # ploting
    x_axis_length = int((rounds - 0.5) // interval) + 1
    x_axis = [i*interval for i in range(1, x_axis_length)]
    x_axis.append(rounds)

    policies = {
        "real_worst": real_worst_generator(arms),
        "ts_best": ts_best,
        "real_second_best": real_second_best_generator(arms)
    }

    for policies_group_name, policies_group in {"ts_bad_policies": policies}.items():
        plt.figure()
        plt.xlabel("rounds")
        plt.ylabel("Changed regrets by max free pull")
        plt.title(f"TS: {trials} trials, {len(arms)} arms")

        # plt.plot(x_axis, default_regrets_diff, label=f"pure_explore")
        for policy_name, policy in policies_group.items():
            regrets_diff, pulling_times = simu.run(base_algo, free_policy=policy,
                                                   interval=interval, times=trials,
                                                   silent=silent, log_pulling_times=log_pulling_times)
            plt.plot(x_axis, regrets_diff, label=f"{policy_name}")
        plt.legend(loc="best")
        plt.savefig(f"./TS_{policies_group_name}.pdf")
        plt.show()

    # for e in epsilon_list:
    #     EpsilonGreedy.epsilon = e
    #     print(f"the epsilon is {EpsilonGreedy.epsilon}")
    #     experiment_algos = [EpsilonGreedy]
    #     for algo in experiment_algos:
    #         simu = Simulator(arms, rounds, 1)
    #         regrets_diff, pulling_times = simu.run(EpsilonGreedy, free_policy=real_second_best_generator(arms),
    #                                                interval=interval, times=trials,
    #                                                silent=silent, log_pulling_times=log_pulling_times)
    #         plt.plot(x_axis, regrets_diff, label=f"{algo.__name__}. epsilon {e}")
