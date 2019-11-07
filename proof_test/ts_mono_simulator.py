from scipy.special import beta
from random import randint
import matplotlib.pyplot as plt
import numpy as np


class BetaComparor:
    def __init__(self):
        pass

    def prob_of_larger(self, a1, b1, a2, b2):
        p = 0
        for i in range (a1):
            p += beta(a2+i, b1+b2)/((b1+i)*beta(1+i, b1)*beta(a2,b2))
        return p

    def simu(self, stepping_param, params=None, param_range=100, steps=100):
        param_name = ["a1", "b1", "a2", "b2"]
        base = 1
        if params is None:
            while True:
                params = [randint(base, base+param_range) for x in range(4)]
                a1, b1, a2, b2 = tuple(params)
                if a1*(a2+b2) > a2*(a1+b1):
                    break
        x_axis = [i for i in range(steps)]
        y = []
        plt.figure()
        ax = plt.gca()
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.xlabel("rounds")
        plt.ylabel("probability of P(u1>u2)")
        plt.title(f"Parameters: {params}. Changed: {param_name[stepping_param]}")
        for i in range(steps):
            y.append(self.prob_of_larger(*params))
            params[stepping_param] += 1

        plt.plot(x_axis, y)
        plt.show()


if __name__ == "__main__":
    # a1+b1 is small, a2+b2 is small
    s1 = [1, 1, 1, 2]
    s2 = [2, 1, 50, 51]
    s3 = [52, 50, 1, 1]
    s4 = [52, 50, 50, 50]
    # if arm1 is unlucky
    s1 = [1, 3, 1, 2]
    s2 = [1, 3, 51, 50]
    s3 = [52, 70, 2, 1]
    s4 = [52, 70, 51, 50]
    simulator = BetaComparor()
    simulator.simu(2, params=s4, param_range=10, steps=50)
