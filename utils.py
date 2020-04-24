import numpy as np
import math

def binary_sampler(r):
    return np.random.choice([0, 1], p=[1 - r, r])


def gaussian_sampler_generator(st_dev):
    def gaussian_sampler(reward):
        r = np.random.normal(reward, st_dev)
        return r
    return gaussian_sampler


def compute_ucb(mean, times, T):
    """Compute upper confidence bound for an arm.
    """
    confidence = math.sqrt(2*math.log(T)/float(times))
    return mean + confidence


def compute_se_ucb(bandit, arm):
    """Compute upper confidence bound for successive elimination strategy.
    """
    if bandit.rejected[arm] == 1: 
        return -1
    mean, times = bandit.normal_pull_record[arm]
    return compute_ucb(mean, times, bandit.T)


def get_second_max(arr):
    copy = arr[:]
    copy.remove(max(arr))
    return arr.index(max(copy))


def get_kth_max(arr, k=1):
    assert not k == 0
    sorted_arr = arr[:]
    sorted_arr.sort(reverse=True)
    sorted_arr.insert(0, 0)
    ele = sorted_arr[k]
    return arr.index(ele)