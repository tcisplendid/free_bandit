from scipy.stats import norm
# from sympy import *
from scipy import integrate
import math
import numpy as np
import matplotlib.pyplot as plt
from decimal import *
getcontext().prec = 8


class EGProb:
    """
    This class computes expected regrets for each round for 2-arm bandits.
    E[regret at t] = P(choose worse arm at t)
    It computes the probability in a dynamic programming way.
    """
    def __init__(self, a1, a2, var):
        # self.a1 = a1
        # self.a2 = a2
        assert a1 > a2
        self.diff = a1 - a2
        self.var = var
        self.p = [0, 0, 1]  # round 0 doesn't exist. round 1 and round 2 are initialization without choice
        # self.comb_p = {(1,1):1}
        self.p_m = [0, 1]
        self.p_free = [0, 0]
        # self.comb_p_free = {(1,1):1}
        self.p_m_free = [0, 1]

    def compute_compared_p(self, m, n):
        return norm.cdf(0, loc=self.diff, scale=(self.var * math.sqrt(1 / m + 1 / n)))

    def compute(self, t):
        def get_comb_p(m, n):
            if (m, n) in self.comb_p.keys():
                return self.comb_p[(m, n)]
            if m == 0 or n == 0:
                return 0
            p_t = get_p(m+n)
            comb_p = get_comb_p(m-1, n)*(1-p_t)+get_comb_p(m,n-1)*p_t
            self.comb_p[(m,n)] = comb_p
            return comb_p

        def update_p_m(t):
            p = get_p(t)
            self.p_m.append(0)
            assert len(self.p_m) == t+1
            for i in range(t, 0, -1):
                self.p_m[i] = self.p_m[i] * p + self.p_m[i-1] * (1-p)
            return

        def get_p(t):
            if t < len(self.p):
                return self.p[t]
            # if t == 3:
            #     p = compute_compared_p(1, 1)
            #     return p
            p = 0
            for i in range(1, t-1):
                # p+=get_comb_p(i, t-i-1)*compute_compared_p(i, t-i-1)
                p += self.p_m[i] * self.compute_compared_p(i, t-i-1)
            self.p.append(p)
            return p

        for i in range(2, t):
            # estimate the probability to choose worse arm at round i
            p = get_p(i)
            # update the probability of choosing best arm X times after round i
            update_p_m(i)
        return p

    def compute_free(self, t):
        def get_comb_p(m, n):
            if (m, n) in self.comb_p_free.keys():
                return self.comb_p_free[(m, n)]
            if m == 0 or n == 0 or m > n:
                return 0
            # t = (m+n)//2
            p_t = get_p((m+n)//2)
            a,b = get_comb_p(m-1, n-1), get_comb_p(m,n-2)
            comb_p = a*(1-p_t)+b*p_t
            # if comb_p == 0:
            print(f"P({m}, {n})={comb_p}={a}*{1-p_t}+{b}*{p_t}. P_{(m + n) // 2}: {p_t}")
            self.comb_p_free[(m,n)] = comb_p
            return comb_p

        def update_p_m(t):
            p = get_p(t)
            self.p_m_free.append(0)
            assert len(self.p_m_free) == t+1
            for i in range(t, 0, -1):
                self.p_m_free[i] = self.p_m_free[i] * p + self.p_m_free[i-1] * (1-p)
            return

        def get_p(t):
            if t < len(self.p_free):
                return self.p_free[t]
            # if t == 3:
            #     p = compute_compared_p(1, 1)
            #     return p
            p = 0
            for i in range(1, t):
                # p+=get_comb_p(i, t-i-1)*compute_compared_p(i, t-i-1)
                p += self.p_m_free[i] * self.compute_compared_p(i, 2*(t-1)-i)
            self.p_free.append(p)
            return p

        for i in range(2, t):
            # estimate the probability to choose worse arm at round i
            p = get_p(i)
            # update the probability of choosing best arm X times after round i
            update_p_m(i)
        return p

        # def get_p(t):
        #     if t < len(self.p_free):
        #         return self.p_free[t]
        #     # if t == 2:
        #     #     p = compute_compared_p(1, 1)
        #     #     return p
        #     p = 0
        #     for i in range(1, t):
        #         m, n = i, 2*(t-1)-i
        #         p += get_comb_p(m, n)*compute_compared_p(m, n)
        #     return p
        #
        # for i in range(2, t):
        #     p = get_p(i)
        #     self.p_free.append(p)
        # return p


class EGProbPrec(EGProb):
    """
    This class does the same job as EGProb but with more precise numbers.
    """
    def __init__(self, a1, a2, var):
        # self.a1 = a1
        # self.a2 = a2
        assert a1 > a2
        self.diff = a1 - a2
        self.var = var
        self.p = [0, 0, 1]  # round 0 doesn't exist. round 1 and round 2 are initialization without choice
        self.comb_p = {(1, 1): 1}
        self.p_free = [0, 0]
        self.comb_p_free = {(1, 1): 1}

    def compute(self, t):
        """
        All functions should return Decimal or int.
        """
        def compute_compared_p(m, n):
            # return norm.cdf(-(self.diff) / (self.var * math.sqrt(1 / m + 1 / n)))
            p = norm.cdf(0, loc=self.diff, scale=(self.var * math.sqrt(1 / m + 1 / n)))
            return Decimal.from_float(p)

        def get_comb_p(m, n):
            if (m, n) in self.comb_p.keys():
                return self.comb_p[(m, n)]
            if m == 0 or n == 0:
                return 0
            p_t = get_p(m+n)
            comb_p = get_comb_p(m-1, n)*(1-p_t)+get_comb_p(m, n-1)*p_t
            self.comb_p[(m, n)] = comb_p
            return comb_p

        def get_p(t):
            if t < len(self.p):
                return self.p[t]
            if t == 3:
                p = compute_compared_p(1, 1)
                return p
            p = 0
            for i in range(1, t-1):
                p += get_comb_p(i, t-i-1)*compute_compared_p(i, t-i-1)
            return p

        for i in range(3, t):
            p = get_p(i)
            self.p.append(p)
        return p

    def compute_free(self, t):
        def compute_compared_p(m, n):
            p = norm.cdf(0, loc=self.diff, scale=(self.var * math.sqrt(1 / m + 1 / n)))
            return Decimal.from_float(p)

        def get_comb_p(m, n):
            if (m, n) in self.comb_p_free.keys():
                return self.comb_p_free[(m, n)]
            if m == 0 or n == 0 or m > n:
                return 0
            # t = (m+n)//2
            p_t = get_p((m+n)//2)
            # a, b = get_comb_p(m-1, n-1), get_comb_p(m, n-2)
            comb_p = get_comb_p(m-1, n-1)*(1-p_t)+get_comb_p(m, n-2)*p_t
            # if comb_p == 0:
            print(f"P({m}, {n})={comb_p}={a}*{1-p_t}+{b}*{p_t}. P_{(m + n) // 2}: {p_t}")
            self.comb_p_free[(m, n)] = comb_p
            return comb_p

        def get_p(t):
            if t < len(self.p_free):
                return self.p_free[t]
            if t == 2:
                p = compute_compared_p(1, 1)
                return p
            p = 0
            for i in range(1, t):
                m, n = i, 2*(t-1)-i
                p += get_comb_p(m, n)*compute_compared_p(m, n)
            return p

        for i in range(2, t):
            p = get_p(i)
            self.p_free.append(p)
        return p


if __name__ == "__main__":
    r_free = [0, -0.001, 0.023, 0.043, 0.062, 0.078, 0.093, 0.106, 0.12, 0.133, 0.145, 0.158, 0.17, 0.184, 0.197, 0.209, 0.218,
     0.226, 0.237, 0.249, 0.261, 0.271, 0.284, 0.296, 0.307, 0.316, 0.327, 0.338, 0.348, 0.357]#, 0.367]
    r_no_free = [0, 0.0, 0.101, 0.124, 0.145, 0.163, 0.179, 0.193, 0.207, 0.22, 0.232, 0.244, 0.256, 0.268, 0.278, 0.289, 0.299,
       0.308, 0.318, 0.33, 0.339, 0.349, 0.359, 0.37, 0.38, 0.389, 0.399, 0.408, 0.418, 0.426] #, 0.437]

    a, b, sigma = 0.5, 0.4, 0.1
    a, b, sigma = 50000, 40000, 10000
    # for i in [a, b, sigma]:
    #     i = i * 1000000
    simu = EGProb(a, b, sigma)
    t = 30
    simu.compute(t)
    simu.compute_free(t)
    plt.figure()
    plt.xlabel("rounds")
    plt.ylabel("prob of choosing suboptimal arm")
    plt.title(f"Numerical computation")

    # diff = Decimal("0.1")
    diff = 0.1
    p = np.cumsum(simu.p)*diff
    p_free = np.cumsum(simu.p_free)*diff
    # p = simu.p
    # p_free = simu.p_free
    print(p)
    print(p_free)
    plt.plot(range(t), p, label=f"no free pull(calculated)")
    plt.plot(range(t), p_free, label=f"free pull with k=1(calculated)")
    plt.plot(range(t), r_no_free, label=f"no free pull(simulated)")
    plt.plot(range(t), r_free, label=f"free pull with k=1(simulated)")
    plt.legend(loc="best")
    plt.show()
    (x, y) = (simu.compute_compared_p(2,2), simu.compute_compared_p(1,3))
    print(x, y, x*0.76+y*0.24)