import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Process

MULTIPLIER = 1
DEFAULT_ST_DEV = 1
DEFAULT_SAMPLE_TIMES = 10000


def get_weighted_mean(values, weights):
    new = [value*weight for value, weight in zip(values, weights)]
    return sum(new)/sum(weights)


def prover_test(a1, a2, m, n, delta_m, delta_n, st_dev=DEFAULT_ST_DEV, test_times=DEFAULT_SAMPLE_TIMES):
    assert a1 >= a2
    for par in (a1, a2, m, n, delta_m, delta_n):
        assert par >= 0
    postive_num_orig = 0
    postive_num_new = 0
    for i in range(test_times):
        estimated_mu1 = np.random.normal(a1, st_dev/math.sqrt(m))
        estimated_mu2 = np.random.normal(a2, st_dev/math.sqrt(n))
        a1_new_signal = np.random.normal(a1, st_dev/math.sqrt(delta_m)) if delta_m > 0 else 0
        a2_new_signal = np.random.normal(a2, st_dev/math.sqrt(delta_n)) if delta_n > 0 else 0
        new_estimated_mu1 = get_weighted_mean([estimated_mu1, a1_new_signal], [m, delta_m])
        new_estimated_mu2 = get_weighted_mean([estimated_mu2, a2_new_signal], [n, delta_n])
        if estimated_mu1 > estimated_mu2:
            postive_num_orig += 1
        if new_estimated_mu1 > new_estimated_mu2:
            postive_num_new += 1
    return (postive_num_new-postive_num_orig)/test_times


def estimate_conditional_prob_for_a1_larger_a2(a1, a2, m, n, delta_m, delta_n,
                                             st_dev=DEFAULT_ST_DEV, test_times=DEFAULT_SAMPLE_TIMES):
    assert a1 >= a2
    for par in (a1, a2, m, n, delta_m, delta_n):
        assert par >= 0
    postive_num_orig = 0
    postive_num_new = 0
    postive_conditional = 0
    for i in range(test_times):
        estimated_mu1 = np.random.normal(a1, st_dev/math.sqrt(m))
        estimated_mu2 = np.random.normal(a2, st_dev/math.sqrt(n))
        a1_new_signal = np.random.normal(a1, st_dev/math.sqrt(delta_m)) if delta_m > 0 else 0
        a2_new_signal = np.random.normal(a2, st_dev/math.sqrt(delta_n)) if delta_n > 0 else 0
        new_estimated_mu1 = get_weighted_mean([estimated_mu1, a1_new_signal], [m, delta_m])
        new_estimated_mu2 = get_weighted_mean([estimated_mu2, a2_new_signal], [n, delta_n])
        if estimated_mu1 > estimated_mu2:
            postive_num_orig += 1
        if new_estimated_mu1 > new_estimated_mu2:
            postive_num_new += 1
    return (postive_num_new-postive_num_orig)/test_times


class EstimatedModel(object):
    def __init__(self, true_means, st_dev, test_times=10000):
        self.arms = true_means
        self.st_dev = st_dev
        self.test_times = test_times
        self.a1, self.a2 = true_means

    def start_test(self, m0, n0, delta_step, rounds):
        s_sqr = 1/m0 + 1/n0
        p0 = 1-norm.cdf((self.a2-self.a1)/(st_dev*math.sqrt(s_sqr)))
        results = [p0]
        m, n = m0, n0
        for i in range(1, rounds+1):
            print(f"compute rounds {i}")
            p = self.estimate_joint_prob(m, n, delta_step)
            m = m + p
            n = n + (1-p)
            results.append(p)
        return results

    def estimate_joint_prob(self, m, n, delta_step):
        st_dev = self.st_dev
        test_times = self.test_times
        a1, a2 = self.a1, self.a2
        p = self.estimate_joint_prob_for_r1_larger_r2(a1, a2, m, n, 1, delta_step, st_dev, test_times) + self.estimate_joint_prob_for_r1_smaller_r2(a1, a2, m, n, 0, delta_step+1, st_dev, test_times)
        return p

    def estimate_joint_prob_for_r1_larger_r2(self, a1, a2, m, n, delta_m, delta_n,
                                             st_dev=None, test_times=None):
        assert a1 >= a2
        for par in (a1, a2, m, n, delta_m, delta_n):
            assert par >= 0
        if st_dev is None:
            st_dev = self.st_dev
        if test_times is None:
            test_times = self.test_times
        postive_joint = 0
        for i in range(test_times):
            estimated_mu1 = np.random.normal(a1, st_dev / math.sqrt(m))
            estimated_mu2 = np.random.normal(a2, st_dev / math.sqrt(n))
            a1_new_signal = np.random.normal(a1, st_dev / math.sqrt(delta_m)) if delta_m > 0 else 0
            a2_new_signal = np.random.normal(a2, st_dev / math.sqrt(delta_n)) if delta_n > 0 else 0
            new_estimated_mu1 = get_weighted_mean([estimated_mu1, a1_new_signal], [m, delta_m])
            new_estimated_mu2 = get_weighted_mean([estimated_mu2, a2_new_signal], [n, delta_n])
            if estimated_mu1 > estimated_mu2 and new_estimated_mu1 > new_estimated_mu2:
                postive_joint += 1
        return postive_joint / test_times

    def estimate_joint_prob_for_r1_smaller_r2(self, a1, a2, m, n, delta_m, delta_n,
                                              st_dev=None, test_times=None):
        assert a1 >= a2
        for par in (a1, a2, m, n, delta_m, delta_n):
            assert par >= 0
        if st_dev is None:
            st_dev = self.st_dev
        if test_times is None:
            test_times = self.test_times
        postive_joint = 0
        for i in range(test_times):
            estimated_mu1 = np.random.normal(a1, st_dev / math.sqrt(m))
            estimated_mu2 = np.random.normal(a2, st_dev / math.sqrt(n))
            a1_new_signal = np.random.normal(a1, st_dev / math.sqrt(delta_m)) if delta_m > 0 else 0
            a2_new_signal = np.random.normal(a2, st_dev / math.sqrt(delta_n)) if delta_n > 0 else 0
            new_estimated_mu1 = get_weighted_mean([estimated_mu1, a1_new_signal], [m, delta_m])
            new_estimated_mu2 = get_weighted_mean([estimated_mu2, a2_new_signal], [n, delta_n])
            if estimated_mu1 <= estimated_mu2 and new_estimated_mu1 > new_estimated_mu2:
                postive_joint += 1
        return postive_joint / test_times
