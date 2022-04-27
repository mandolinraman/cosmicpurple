"speed tests for logsuemxp"

import functools
import time
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from numba import jit


@jit
def softmax(path_metric):
    """
    Compute the soft max operation of all incoming pathmetrics at each state
    """

    state_metric = -np.inf
    gamma = 0.0
    for metric in path_metric:
        delta = state_metric - metric
        if np.isnan(delta):
            gamma += 1.0
        elif delta < 0:
            gamma = gamma * np.exp(delta) + 1.0
            state_metric = metric
        else:
            gamma += np.exp(-delta)

    state_metric += np.log(gamma)  # softMax operation

    return state_metric, gamma


# lambda function, couldn't jit it?
# softmax_step_tuple = (
#     lambda t, u: (t[0], t[1] + np.exp(u - t[0]))
#     if t[0] > u
#     else (u, t[1] * np.exp(t[0] - u) + 1)
# )


@jit
def softmax_step_tuple(t, u):
    "softmax step tuple version"
    return (
        (t[0], t[1] + np.exp(u - t[0]))
        if t[0] > u
        else (u, t[1] * np.exp(t[0] - u) + 1)
    )


@jit
def softmax_step_tuple_alt(t, u):
    "softmax step tuple version"
    delta = t[0] - u
    return (
        (t[0], t[1] + np.exp(-delta))
        if delta > 0
        else (u, t[1] * np.exp(delta) + 1.0)
    )


@jit
def softmax_step_inplace(t, u):
    "softmax step inplace version"
    delta = t[0] - u
    if delta > 0:
        t[1] += np.exp(-delta)
    else:
        t[0] = u
        t[1] = t[1] * np.exp(delta) + 1.0
    return t


N = 10000
a = np.random.rand(N)
names = [
    "logsumexp",
    "logaddexp.reduce",
    "softmax",
    "sm-tuple",
    "sm-tuple-alt",
    "sm-inplace",
]

times = [[] for method in range(len(names))]

for p in range(10):
    method = 0

    print(f"Pass: {1 + p}")
    t0 = time.time()
    b = logsumexp(a)
    t1 = time.time()
    print(f"names[method]: {b}")
    print(f"Time = {t1-t0} seconds\n")
    times[method].append(t1 - t0)

    method += 1
    t0 = time.time()
    b = np.logaddexp.reduce(a)
    t1 = time.time()
    print(f"names[method]: {b}")
    print(f"Time = {t1-t0} seconds\n")
    times[method].append(t1 - t0)

    method += 1
    t0 = time.time()
    b = softmax(a)
    t1 = time.time()
    print(f"names[method]: {b}")
    print(f"Time = {t1-t0} seconds\n")
    times[method].append(t1 - t0)

    method += 1
    t0 = time.time()
    t = functools.reduce(softmax_step_tuple, a, (-np.inf, 0))
    b = t[0] + np.log(t[1])
    t1 = time.time()
    print(f"names[method]: {b}")
    print(f"Time = {t1-t0} seconds\n")
    times[method].append(t1 - t0)

    method += 1
    t0 = time.time()
    t = functools.reduce(softmax_step_tuple_alt, a, (-np.inf, 0))
    b = t[0] + np.log(t[1])
    t1 = time.time()
    print(f"names[method]: {b}")
    print(f"Time = {t1-t0} seconds\n")
    times[method].append(t1 - t0)

    method += 1
    t0 = time.time()
    t = np.array([-np.inf, 0])
    functools.reduce(softmax_step_inplace, a, t)
    b = t[0] + np.log(t[1])
    t1 = time.time()
    print(f"names[method]: {b}")
    print(f"Time = {t1-t0} seconds\n")
    times[method].append(t1 - t0)

    method += 1

for method, name in enumerate(names):
    plt.semilogy(times[method], label=f"{name}")
    plt.legend()

plt.grid()
plt.show()
