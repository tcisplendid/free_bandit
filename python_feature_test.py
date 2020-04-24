import numpy as np
import math
import matplotlib.pyplot as plt
# from scipy.stats import norm
from datetime import date
from decimal import *


if __name__ == '__main__':
    def confidence(t, at):
        return math.sqrt(2 * math.log(t) / float(at))
    print(confidence(8,6))
