import numpy as np


def crowding_distance_assignment(I):

    ### algorithm 3 from Deb et al. (2002)

    l = len(I)  # number of solutions
    N_obj = len(I[0])  # number of objectives

    d = np.zeros(l)  # crowding distance of each solution

    if np.any(np.isinf(I)):
        raise Exception("problem!")

    for m in range(N_obj):
        idx_m = np.argsort(I[:, m])[::-1]
        # I = sorted(I, key=lambda x: x[m])  # sort the solutions to obj. m
        d[idx_m[0]] += np.inf  # set first and ...
        d[idx_m[-1]] += np.inf  # ... last solution to infinity
        for i in range(1, l - 1):
            d[idx_m[i]] += (I[idx_m[i + 1]][m] - I[idx_m[i - 1]][m]) / (
                I[idx_m[l - 1]][m] - I[idx_m[0]][m]
            )  # compute the crowding distance

    return d  # return the crowding distance of each solution
