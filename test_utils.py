import numpy as np
from utils import *


num = 10000
num_buckets = 5000
buckets = np.random.permutation(
    np.tile(range(num_buckets), num // num_buckets)
)
# buckets = np.random.randint(0, num_buckets, num)

metrics = np.random.rand(num)

# single reducer
red0 = Reducer(num, temperature=0)
red1 = Reducer(num, temperature=1)

b0, i0 = red0.reduce(metrics)
assert i0 == np.argmax(metrics)
print(b0 == np.max(metrics))

b1 = red1.reduce(metrics)
print(b1 - logsumexp(metrics))

## multi reducer
multired0 = MultiReducer(num_buckets, buckets, temperature=0)
multired1 = MultiReducer(num_buckets, buckets, temperature=1)

locations = [np.where(buckets == i)[0] for i in range(num_buckets)]

b0 = np.zeros(num_buckets)
multired0.reduce(metrics, output=b0)
b0a = np.array([np.max(metrics[locations[i]]) for i in range(num_buckets)])
b0b = np.array(
    [np.max([metrics[j] for j in locations[i]]) for i in range(num_buckets)]
)

print(np.linalg.norm(b0 - b0a))
print(np.linalg.norm(b0 - b0b))

# soft reduce
b1 = np.zeros(num_buckets)
multired1.reduce(metrics, output=b1)
b1a = np.array([logsumexp(metrics[locations[i]]) for i in range(num_buckets)])
b1b = np.array(
    [logsumexp([metrics[j] for j in locations[i]]) for i in range(num_buckets)]
)

print(np.linalg.norm(b1 - b1a))
print(np.linalg.norm(b1 - b1b))
