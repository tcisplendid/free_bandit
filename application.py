from experiment_base import *
from strategies import *
from scipy.stats import beta


class TempScheduler(Device):
    """
    This class accepts an optimal temperature scheduler as true rewards.
    In fact, it accepts a population of utility function.
    1. define several parameters which decide the utility function, if the utility function can be described
    with several parameters.
    2. define the distribution of those parameters. It gives us population generator.
    3. each pull will sample from population generator n times. n can be sampled from a distribution too.

    Note: the utility function may consider outdoor environment, because sometimes indoor temperature can't
    decide the comfort. For example, in winter, the locations near windows will be warmer if sunshine is strong
    or colder if wind is strong. However, we don't consider outdoor environment for now. It means a person has
    a fixed preferred temp for the whole day.

    Note 2: arms are from [60'F, 76'F] according to https://en.wikipedia.org/wiki/Room_temperature#Comfort_temperatures
    68 is the optimal. A bit scaling of the interval is for computation convenience.

    Note 3: we assume a linear utility_func for simplicity
    """

    def __init__(self, utility_func=None, popu_gen=None, arms=None, n=100, *popu_args):
        """
        Args:
            utility_func: func(args)->utility
            popu_gen: func(*popu_args)->args
            *popu_args: arguments taken by popu_gen
        """
        def u_func(t, preference):
            return 1 - abs(t - preference) / (upper - low)

        self.const = (upper, low, best) = 76, 60, 68
        self.u_func = u_func
        self.n = n
        self.popu_args = popu_args
        if arms is None:
            self.arms = [self.__get_estimated_reward__(50000, t) for t in range(low, upper+1)]
        else:
            self.arms = arms
        self.best_arm_index = best - low
        self.best_arm_val = self.arms[self.best_arm_index]

    def pull(self, arm):
        upper, low, best = self.const
        chosen_temp = low + arm
        r = self.__get_estimated_reward__(1, chosen_temp, 0.1)
        # print("reward is: ", r, "regret is: ", self.best_arm_val - r)
        return r
        #
        # total_reward = 0
        # popu = int(np.random.normal(1, 0.1)*self.n)
        # for i in range(popu):
        #     pref = int(np.random.normal(best, best-low-2))
        #     total_reward += self.u_func(chosen_temp, pref)
        # print(chosen_temp, popu, total_reward/popu)
        # return total_reward/popu

    def __get_estimated_reward__(self, times, temp, st_dev=0):
        total_total_reward = 0
        upper, low, best = self.const
        chosen_temp = temp
        popu = max(int(np.random.normal(1, st_dev)*self.n), 1)
        # popu = self.n
        for i in range(times):
            total_reward = 0
            for i in range(popu):
                # choose best+0.5 because 68 is best but int(normal(68))
                # can return 67 and 68 equally.
                pref = int(np.random.normal(best+0.5, best - low - 2))
                total_reward += self.u_func(chosen_temp, pref)
            total_total_reward += total_reward / popu
        return total_total_reward / times

    def estimate_reward(self):
        pass


def beta_shape_reward_generator(l, a, b, loc=0, scale=1):
    # arr = [i / l for i in range(l+1)]
    res = [beta.cdf((i+1)/l, a, b) - beta.cdf(i/l, a, b) for i in range(l)]
    return [x/scale + loc for x in res]


class SimpleDevice(Device):
    variance = 1

    def pull(self, arm):
        r = self.arms[arm]
        if arm == 0:
            return r
        else:
            return np.random.choice([self.variance, -self.variance], p=[0.5, 0.5]) + r


if __name__ == "__main__":
    # for _ in range(1):
    #     a = TempScheduler(n=1)
    #     print(a.arms)
    #     print(real_second_best_generator(a.arms))
    r = 0
    d = SimpleDevice([0.5, 0])
    for _ in range(15):
        print(d.pull(0), d.pull(1))