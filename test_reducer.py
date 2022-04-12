import numpy as np
from scipy.special import logsumexp
from utils import *


num = 10000
num_buckets = 5000
buckets = np.random.permutation(np.tile(range(num_buckets), num//num_buckets))
# buckets = np.random.randint(0, num_buckets, num)

metrics = np.random.rand(num)
red = Reducer(num)

b0 = red.reduce(metrics)
b1 = red.soft_reduce(metrics)

# reduce
multi_red = MultiReducer(num_buckets, buckets)
locations = [np.where(buckets == i)[0] for i in range(num_buckets)]

b0 = np.zeros(num_buckets)
%timeit multi_red.reduce(metrics, b0)
%timeit b0 = [np.max(metrics[locations[i]]) for i in range(num_buckets)]
%timeit b0 = [np.max(metrics[j] for j in locations[i]) for i in range(num_buckets)]
%timeit b0 = [np.max([metrics[j] for j in locations[i]]) for i in range(num_buckets)]

# soft reduce
b1 = np.zeros(num_buckets)
%timeit multi_red.soft_reduce(metrics, b1)
%timeit b1 = [logsumexp(metrics[locations[i]]) for i in range(num_buckets)]
# %timeit b1 = [logsumexp(metrics[j] for j in locations[i]) for i in range(num_buckets)]
%timeit b1 = [logsumexp([metrics[j] for j in locations[i]]) for i in range(num_buckets)]
