from scipy.stats import norm
# from sympy import *
from scipy import integrate
import math
import numpy as np
import matplotlib.pyplot as plt


class EGProb:
    def __init__(self, a1, a2, var):
        # self.a1 = a1
        # self.a2 = a2
        assert a1 > a2
        self.diff = a1 - a2
        self.var = var
        self.p = [0, 0, 1]  # round 0 doesn't exist. round 1 and round 2 are initialization without choice
        self.comb_p = {(1,1):1}
        self.p_free = [0, 0]
        self.comb_p_free = {(1,3):1}

    def compute(self, t):
        def compute_compared_p(m, n):
            return norm.cdf(-(self.diff) / (self.var * math.sqrt(1 / m + 1 / n)))

        def get_comb_p(m, n):
            if (m, n) in self.comb_p.keys():
                return self.comb_p[(m, n)]
            if m == 0 or n == 0:
                return 0
            p_t = get_p(m+n)
            comb_p = get_comb_p(m-1, n)*(1-p_t)+get_comb_p(m,n-1)*p_t
            self.comb_p[(m,n)] = comb_p
            return comb_p

        def get_p(t):
            if t < len(self.p):
                return self.p[t]
            if t == 3:
                p = compute_compared_p(1, 1)
                return p
            p = 0
            for i in range(1, t-1):
                p+=get_comb_p(i, t-i-1)*compute_compared_p(i, t-i-1)
            return p

        for i in range(3, t):
            p = get_p(i)
            self.p.append(p)
        return p

    def compute_free(self, t):
        def compute_compared_p(m, n):
            return norm.cdf(-(self.diff) / (self.var * math.sqrt(1 / m + 1 / n)))

        def get_comb_p(m, n):
            if (m, n) in self.comb_p_free.keys():
                return self.comb_p_free[(m, n)]
            if m == 0 or n == 0 or m == n:
                return 0
            t = (m+n)//2
            p_t = get_p((m+n)//2)
            comb_p = get_comb_p(m-1, n-1)*(1-p_t)+get_comb_p(m,n-2)*p_t
            self.comb_p_free[(m,n)] = comb_p
            return comb_p

        def get_p(t):
            if t < len(self.p_free):
                return self.p_free[t]
            if t == 2:
                p = compute_compared_p(1, 1)
                return p
            p = 0
            for i in range(1, t-1):
                m, n = i, 2*(t-1)-i
                p+=get_comb_p(m, n)*compute_compared_p(m, n)
            return p

        for i in range(2, t):
            p = get_p(i)
            self.p_free.append(p)
        return p


def integral_test(a, b, sigma):
    def p_a_less_b_func(B, A):
        return norm.pdf(B, loc=b, scale=sigma)*norm.pdf(A, loc=a, scale=sigma)

    # p_a_less_b = norm.cdf(-(a-b) / (sigma * math.sqrt(2)))
    p0 = norm.cdf(0, loc=a - b, scale=math.sqrt(2) * sigma)
    p_a_less_b2 = integrate.dblquad(p_a_less_b_func, -np.inf, np.inf, lambda y: y, np.inf)[0]
    print(p0, p_a_less_b2)

def integral_test2(a, b, sigma):
    def condition_prob(mu1, mu2, oppo=False):
        '''
        mu1 is the mu of the arm A, which is chosen at round 1
        mu2 is for another arm B.
        A0 is inner integral variable
        B0 is outer integral variable
        1. Given A0, B0, A1 is r.v.,
        (1) defaultly compute the conditional prob that A is chosen at round 1, B is chosen at round 2
        P((A0+A1)/2 < B0 | A0>B0)
        (2) if oppo is set True, compute the conditional prob that B is chosen at both rounds
        P((A0+A1)/2 > B0 | A0>B0)
        2. Essentially this functions computes a double integral ∫∫P((A0+A1)/2<B0)P(A0)P(B0)dA0dB0
        The inner integral is [B0, +oo) to ensure the condition [A0 > B0]
        P(A0) = norm.pdf(mu1, sigma) P(B0) similar
        P((A0+A1)/2<B0) = P(A1 < 2B0 - A0) = norm.cdf
        '''

        # x = symbols("x", real=True)
        # y = symbols("y", real=True)
        def p_less(A0, B0):
            p_A_less_B = norm.cdf(2 * B0 - A0, loc=mu1, scale=sigma)
            p_A0 = norm.pdf(A0, loc=mu1, scale=sigma)
            p_B0 = norm.pdf(B0, loc=mu2, scale=sigma)
            return p_A_less_B * p_A0 * p_B0

        def p_greater(x, y):
            p_A_greater_B = 1 - norm.cdf(2 * y - x, loc=mu1, scale=sigma)
            p_A0 = norm.pdf(x, loc=mu1, scale=sigma)
            p_B0 = norm.pdf(y, loc=mu2, scale=sigma)
            return p_A_greater_B * p_A0 * p_B0

        if oppo:
            res = integrate.dblquad(p_greater, -np.inf, np.inf, lambda y: y, np.inf)
        else:
            res = integrate.dblquad(p_less, -np.inf, np.inf, lambda y: y, np.inf)
        print(res)
        return res[0]

    p0 = norm.cdf(0, loc=a - b, scale=math.sqrt(2) * sigma)
    p1 = condition_prob(a, b)  # P((A0+A1)/2 < B0 | A0>B0)
    p2 = condition_prob(b, a)  # P((B0+B1)/2 < A0 | B0>A0)
    p3 = condition_prob(b, a, True) # # P((B0+B1)/2 > A0 | B0>A0)
    p4 = condition_prob(a, b, True)
    print(p1+p2+p3+p4)
    # times = 100000
    # p, sum = 0, 0
    #
    # for i in range(times):
    #     A0 = np.random.normal(a, sigma)
    #     B0 = np.random.normal(b, sigma)
    #     # if A0 < B0:
    #     #     sum += 1
    #     #     B1 = np.random.normal(b, sigma)
    #     #     if (B0 + B1) / 2 > A0:
    #     #         p += 1
    #     if A0 > B0:
    #         sum += 1
    #         A1 = np.random.normal(a, sigma)
    #         if (A0 + A1) / 2 < B0:
    #             p += 1
    # print(p1, p/times, p/sum, p/sum*(1-p0))

def compute_round_2(a, b, sigma):
    def condition_prob(mu1,mu2,sigma=sigma, oppo=False, scale_arm=0):
        '''
        mu1 is the mu of the arm A, which is chosen at round 1
        mu2 is for another arm B.
        A0 is inner integral variable
        B0 is outer integral variable
        1. Given A0, B0, A1 is r.v.,
        (1) defaultly compute the conditional prob that A is chosen at round 1, B is chosen at round 2
        P((A0+A1)/2 < B0 | A0>B0)
        (2) if oppo is set True, compute the conditional prob that B is chosen at both rounds
        P((A0+A1)/2 > B0 | A0>B0)
        2. Essentially this functions computes a double integral ∫∫P((A0+A1)/2<B0)P(A0)P(B0)dA0dB0
        The inner integral is [B0, +oo) to ensure the condition [A0 > B0]
        P(A0) = norm.pdf(mu1, sigma) P(B0) similar
        P((A0+A1)/2<B0) = P(A1 < 2B0 - A0) = norm.cdf
        '''
        # x = symbols("x", real=True)
        # y = symbols("y", real=True)
        if scale_arm == 2:
            sigma = [sigma, sigma/math.sqrt(2)]
        elif scale_arm == 1:
            sigma = [sigma/math.sqrt(2), sigma]
        else:
            sigma = [sigma, sigma]
        def p_less(A0, B0):
            p_A_less_B = norm.cdf(2 * B0 - A0, loc=mu1, scale=sigma[0])
            p_A0 = norm.pdf(A0, loc=mu1, scale=sigma[0])
            p_B0 = norm.pdf(B0, loc=mu2, scale=sigma[1])
            return p_A_less_B * p_A0 * p_B0
        def p_greater(x, y):
            p_A_greater_B = 1 - norm.cdf(2 * y - x, loc=mu1, scale=sigma[0])
            p_A0 = norm.pdf(x, loc=mu1, scale=sigma[0])
            p_B0 = norm.pdf(y, loc=mu2, scale=sigma[1])
            return p_A_greater_B * p_A0 * p_B0
        if oppo:
            res = integrate.dblquad(p_greater, -np.inf, np.inf, lambda y: y, np.inf)
        else:
            res = integrate.dblquad(p_less, -np.inf, np.inf, lambda y: y, np.inf)
        print(res)
        return res[0]

    # np.random.normal(a, sigma)

    # compute p(A0<B0)
    # p0 = norm.cdf(0, loc=a - b, scale=math.sqrt(2) * sigma)
    p_A1B1 = condition_prob(a, b)  # P((A0+A1)/2 < B0 | A0>B0)
    p_B1A1 = condition_prob(b, a)  # P((B0+B1)/2 < A0 | B0>A0)
    p_B1B2 = condition_prob(b, a, oppo=True)  # p0 * (1-p2)
    # p_A1A2 = condition_prob(a, b, oppo=True)  # (1-p1)

    # print(f"p0: {p0}, p_A1B1: {p_A1B1}, p_B1A1: {p_B1A1}, p_B1B2: {p_B1B2}, p_A1A2: {p_A1A2}, sum: {p_B1B2+p_B1A1+p_A1B1+p_A1A2}")
    res = p_A1B1 + p_B1A1 + 2 * p_B1B2
    print(f"regret by history:{res}, regret by expectation: ")


def compute_round2_by_simulation(a, b, sigma, times=100000):
    A1A2, A1B1, B1A1, B1B2 = 0, 1, 2, 3
    p = [0 for i in range(4)]
    for i in range(times):
        A0 = np.random.normal(a, sigma)
        B0 = np.random.normal(b, sigma)
        if A0 > B0:
            A1 = np.random.normal(a, sigma)
            if (A0+A1)/2 > B0:
                p[A1A2] += 1
            else:
                p[A1B1] += 1
        else:
            B1 = np.random.normal(b, sigma)
            if A0 > (B0+B1)/2:
                p[B1A1] += 1
            else:
                p[B1B2] += 1
    p = [x/times for x in p]
    print(p)
    print(p[A1B1]+p[B1A1]+2*p[B1B2])

def compute_round3_by_simulation(a, b, sigma, times=1000000):
    A1A2A3, A1A2B1, A1B1A2, A1B1B2, B1A1A2, B1A1B2, B1B2A1, B1B2B3 = 0, 1, 2, 3, 4, 5, 6, 7
    p = [0 for i in range(8)]
    B_1, B_2, B_3 = 0, 1, 2
    p2 = [0 for i in range(3)]
    for i in range(times):
        A0 = np.random.normal(a, sigma)
        B0 = np.random.normal(b, sigma)
        if A0 > B0:
            A1 = np.random.normal(a, sigma)
            if (A0+A1)/2 > B0:
                A2 = np.random.normal(a, sigma)
                if (A0 + A1 + A2) / 3 > B0:
                    p[A1A2A3] += 1
                else:
                    p2[B_3] += 1
                    p[A1A2B1] += 1
            else:
                p2[B_2] += 1
                B1 = np.random.normal(b, sigma)
                if (A0 + A1)/2 > (B0+B1)/2:
                    p[A1B1A2] += 1
                else:
                    p2[B_3] += 1
                    p[A1B1B2] += 1
        else:
            p2[B_1] += 1
            B1 = np.random.normal(b, sigma)
            if A0 > (B0+B1)/2:
                A1 = np.random.normal(a, sigma)
                if (A0 + A1)/2 > (B0+B1)/2:
                    p[B1A1A2] += 1
                else:
                    p2[B_3] += 1
                    p[B1A1B2] += 1
            else:
                p2[B_2] += 1
                B2 = np.random.normal(b, sigma)
                if A0 > (B0+B1+B2)/3:
                    p[B1B2A1] += 1
                else:
                    p2[B_3] += 1
                    p[B1B2B3] += 1
    p = [x/times for x in p]
    print(p)
    print(p[A1A2B1]+p[A1B1A2]+p[B1A1A2]+2*(p[A1B1B2]+p[B1A1B2]+p[A1B1B2])+3*p[B1B2B3])
    p2 = [x / times for x in p2]
    print(p2)
    print(p2[B_1] + p2[B_2] + p2[B_3])

def compute_round2_seperately_by_simulation(a, b, sigma, times=100000):
    B_1, B_2 = 0, 1
    p = [0 for i in range(2)]
    for i in range(times):
        A0 = np.random.normal(a, sigma)
        B0 = np.random.normal(b, sigma)
        if A0 > B0:
            A1 = np.random.normal(a, sigma)
            if (A0+A1)/2 < B0:
                p[B_2]+=1
        else:
            p[B_1] += 1
            B1 = np.random.normal(b, sigma)
            if A0 < (B0+B1)/2:
                p[B_2] += 1
    p = [x/times for x in p]
    print(p)
    print(p[B_1]+p[B_2])

def compute_round3_seperately_by_simulation(a, b, sigma, times=100000):
    B_1, B_2, B_3= 0, 1, 2
    p = [0 for i in range(3)]
    for i in range(times):
        A0 = np.random.normal(a, sigma)
        B0 = np.random.normal(b, sigma)
        if A0 > B0:
            A1 = np.random.normal(a, sigma)
            if (A0+A1)/2 > B0:
                A2 = np.random.normal(a, sigma)
                if (A0 + A1 + A2) / 3 < B0:
                    p[B_3] += 1
            else:
                B1 = np.random.normal(b, sigma)
                p[B_2] += 1
                if (A0 + A1)/2 < (B0+B1)/2:
                    p[B_3] += 1
        else:
            p[B_1] += 1
            B1 = np.random.normal(b, sigma)
            if A0 > (B0+B1)/2:
                A1 = np.random.normal(a, sigma)
                if (A0 + A1)/2 < (B0+B1)/2:
                    p[B_3] += 1
            else:
                p[B_2] += 1
                B2 = np.random.normal(b, sigma)
                if A0 < (B0+B1+B2)/3:
                    p[B_3] += 1

    p = [x/times for x in p]
    print(p)
    print(p[B_1]+p[B_2]+p[B_3])


def compute_round1_by_simulation(a, b, sigma, times=100000):
    p = 0
    for i in range(times):
        A0 = np.random.normal(a, sigma)
        B0 = np.random.normal(b, sigma)
        if A0 < B0:
            p += 1
    p = p/times
    print(p)


if __name__ == "__main__":
    simu = EGProb(0.5, 0.4, 0.1)
    t = 300
    simu.compute(t)
    simu.compute_free(t)
    plt.figure()
    plt.xlabel("rounds")
    plt.ylabel("prob of choosing suboptimal arm")
    plt.title(f"Numerical computation")

    p = np.cumsum(simu.p)
    p_free = np.cumsum(simu.p_free)
    print(p)
    print(p_free)
    plt.plot(range(t), p, label=f"no free pull")
    plt.plot(range(t), p_free, label=f"free pull with k=1")
    plt.legend(loc="best")
    plt.show()
    # compute_round_2(5,4,1)
    # a, b, sigma, times = 0.5, 0.4, 0.1, 1000000
    # # compute_round3_by_simulation(a, b, sigma, times)
    # # compute_round3_seperately_by_simulation(a, b, sigma, times)
    # compute_round_2(a, b, sigma)
    # simu = EGProb(a, b, sigma)
    # simu.compute(5)
    # p = simu.p
    # # print(p, p[3]+p[4])
    # p0 = norm.cdf(0, loc=a - b, scale=math.sqrt(2) * sigma)
    # print(p0)
    # compute_round1_by_simulation(a, b, sigma, times)