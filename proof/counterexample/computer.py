import matplotlib.pyplot as plt
import math
import numpy as np

def compute_regret_no_free(t, val):
    def find_positive(arr):
        x = [x for x in arr if x > 0]
        return len(x)

    if t == 1:
        return [0]
    if t == 2:
        return [0, 1]
    reg = [0, 1]
    possible_seq = [[]]
    for i in range(3, t+1):
        tmp = []
        for seq in possible_seq:
            print(seq)
            x = find_positive(seq + [-1])
            y = i-2-x
            if x - y > (i-2) * val:
                tmp.append(seq+[-1])
            seq.append(1)
        possible_seq.extend(tmp)
        print(i, possible_seq)
        reg.append(len(possible_seq)/2**(i-2))
    return reg


# def false_compute_reg_free(t, val):
#     def find_positive(arr):
#         x = [x for x in arr if x > 0]
#         return len(x)
#
#     if t == 1:
#         return [0]
#     if t == 2:
#         return [0, 0.5]
#     reg = [0, 0.5]
#     possible_seq = [[1]]
#     signals = [[-1,-1], [-1, 1], [1, -1]]
#     for i in range(3, t+1):
#         tmp = []
#         for seq in possible_seq:
#             # print(seq)
#             signal_num = 2*(i-1)-1
#             x = find_positive(seq + [-1, -1])
#             y = signal_num-x
#             if x - y > signal_num * val:
#                 tmp.append(seq+[-1, -1])
#             x = find_positive(seq + [-1, 1])
#             y = signal_num-x
#             if x - y > signal_num * val:
#                 tmp.append(seq + [-1, 1])
#                 tmp.append(seq + [1, -1])
#             seq.append(1)
#             seq.append(1)
#         possible_seq.extend(tmp)
#         print(i)
#         print(possible_seq)
#         reg.append(len(possible_seq)/(2*4**(i-2)))
#     return reg


def compute_regret_free(t, val):
    def find_num(seq, tar):
        sum = 0
        for e in seq:
            if e == tar:
                sum += 1
        return sum

    def validate_seq(seq, val):
        sum, times = 0, 0
        for i in range(len(seq)-2):
            if not seq[i] == 0:
                sum += seq[i]
                times += 1
        mean = sum / times
        if seq[-2] == 0:
            return mean <= val
        else:
            return mean > val

    def find_ans(seq, depth):
        if depth == 0:
            # n is the number of choosing the best arm
            # the number is the weight of this possible event
            # in the final probability computation
            sum, times = 0, 0
            for e in seq:
                if not e == 0:
                    sum += e
                    times += 1
            mean = sum / times
            if mean > val:
                # print(seq)
                return 2**find_num(seq, 0)
            else:
                return 0
        possible_seqs = [[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 1], [0, -1]]
        ans = 0
        for s in possible_seqs:
            s = seq + s
            if validate_seq(s, val):
                ans += find_ans(s, depth-1)
        return ans

    if t == 1:
        return [0]
    if t == 2:
        return [0, 1]
    # if t == 3:
    #     return [0, 1, 0.25]
    # reg = [0, 1, 0.25]
    reg = [0, 1]
    possible_seqs = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    for depth in range(3-3, t+1-3):
        print(depth+3)
        ans = 0
        for seq in possible_seqs:
            ans += find_ans(seq, depth)
        reg.append(ans/4**(depth+1))
    return reg




if __name__ == "__main__":
    # regrets k=1: [0.0, 0.046, 0.081, 0.129, 0.165, 0.218, 0.257, 0.292, 0.327, 0.375]
    # regrets k=no free pull: [0.0, 0.098, 0.167, 0.197, 0.218, 0.236, 0.259, 0.271, 0.282, 0.295]
    # regrets k=1: [0.0, 0.047, 0.094, 0.143, 0.163, 0.217, 0.248, 0.276, 0.301, 0.336]
    # regrets k=no free pull: [0.0, 0.082, 0.134, 0.163, 0.175, 0.193, 0.211, 0.226, 0.243, 0.259]

    # [0.         0.1        0.14375    0.18125    0.22382813 0.26328125
    #  0.29897461 0.33114014 0.36336517 0.39354477 0.4243288 ]
    # [0.         0.1        0.15       0.175      0.2        0.21875
    #  0.2375     0.253125   0.26875    0.28242187 0.29609375]
    # regrets k=1: [0.0, 0.104, 0.121, 0.161, 0.194, 0.235, 0.268, 0.309, 0.344, 0.386]
    # #regrets k=no free pull: [0.0, 0.103, 0.154, 0.18, 0.203, 0.222, 0.243, 0.258, 0.283, 0.294]

    t = 11
    val = diff = 0.2
    # print(compute_regret_free(4, val))
    b = compute_regret_free(t, val)
    a = compute_regret_no_free(t, val)
    plt.figure()
    plt.xlabel("rounds")
    plt.ylabel("prob of choosing suboptimal arm")
    plt.title(f"Numerical computation for [{val}, 0]")

    p = np.cumsum(a) * diff
    p_free = np.cumsum(b) * diff
    print(p_free)
    print(p)
    plt.plot(range(t), p_free, label=f"free pull with k=1(calculated)")
    plt.plot(range(t), p, label=f"no free pull(calculated)")
    # plt.plot(range(t), r_no_free, label=f"no free pull(simulated)")
    # plt.plot(range(t), r_free, label=f"free pull with k=1(simulated)")
    plt.legend(loc="best")
    plt.show()
