"utils.py tests"
import numpy as np
from scipy.special import logsumexp
from cosmicpurple.utils import Reducer, MultiReducer, aggregate


num = 10000
num_buckets = 5000
buckets = np.random.permutation(
    np.tile(range(num_buckets), num // num_buckets)
)
# buckets = np.random.randint(0, num_buckets, num)

metrics = np.random.rand(num)

# aggregate function
result = aggregate(metrics, num_buckets, buckets)


# single reducer
red0 = Reducer(num, temperature=0)
b0, i0 = red0.reduce(metrics)
print(b0 - np.max(metrics))  # 0

red1 = Reducer(num, temperature=1)
b1 = red1.reduce(metrics, compute_softmax=True)
print(b1 - logsumexp(metrics))  # 0

## multi reducer
multired0 = MultiReducer(num_buckets, buckets, temperature=0)
b0 = np.zeros(num_buckets)
multired0.reduce(metrics, output=b0)
# preprocess the buckets
locations = [np.where(buckets == i)[0] for i in range(num_buckets)]
b0_alt = np.array([np.max(metrics[locations[i]]) for i in range(num_buckets)])
print(np.linalg.norm(b0 - b0_alt))  # 0

# soft reduce
multired1 = MultiReducer(num_buckets, buckets, temperature=1)
b1 = np.zeros(num_buckets)
multired1.reduce(metrics, output=b1, compute_softmax=True)
b1_alt = np.array(
    [logsumexp(metrics[locations[i]]) for i in range(num_buckets)]
)
print(np.linalg.norm(b1 - b1_alt))  # 0
