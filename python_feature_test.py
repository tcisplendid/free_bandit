import numpy as np
import math
import matplotlib.pyplot as plt
# from scipy.stats import norm
from datetime import date


class Test(object):
    def __init__(self):
        pass

    def test(self, check):
        if check:
            def say():
                print("check is True")
        else:
            def say():
                print("check is False")
        say()


if __name__ == '__main__':
    a = {
        "a":1,
        "c":2,
        "b":3
    }
    t = Test()
    t.test(False)

    data = [
        13452222,
        27621337,
        5284754,
        4124037
    ]
    print(sum(data))
