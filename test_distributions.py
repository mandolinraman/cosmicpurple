import numpy as np
import math
import distributions as dist
from pomegranate import *


def check_equal(a, b):
    print(a)
    print(b)
    if all(math.isclose(ai, bi) for (ai, bi) in zip(a, b)):
        print("Passed!")
    else:
        print("Failed.")


num_samples = 100000
weights = np.random.rand(num_samples)

# Discrete distribution
print("\nDiscrete distribution")
data = np.floor(4 * np.random.rand(num_samples) ** 2).astype(int)

d0 = dist.DiscreteDistribution.from_samples(data, 4, weights=weights)
d1 = DiscreteDistribution.from_samples(data, weights=weights)
for i in range(4):
    print(f"{i}: ", d0.probability(i), d1.probability(i))

check_equal(
    [d0.probability(i) for i in range(4)],
    [d1.probability(i) for i in range(4)],
)

# Normal distribution
print("\nNormal distribution")
data = np.random.randn(num_samples) * 2 + 5
d0 = dist.GaussianDistribution.from_samples(data, weights=weights)
d1 = NormalDistribution.from_samples(data, weights=weights)
print("d0: [mean, std]", [d0.mean, np.sqrt(d0.var)])
print("d1: [mean, std]", d1.parameters)
check_equal([d0.mean, np.sqrt(d0.var)], d1.parameters)

# AR Normal distribution
print("\nAR Normal distribution")
predf = np.array([0.5, -0.6, 0.9, 0.3])
data = np.random.randn(num_samples, 5)
bias, sigma = 3, 2
data[:, 0] = bias + data[:, 0] * sigma + data[:, 1:] @ predf
d0 = dist.AutoregressiveGaussianDistribution.from_samples(
    data, weights=weights
)

print(f"{d0.mean} must be close to {bias}")
print(f"{d0.var} must be close to {sigma**2}")
print(f"{-d0.whitener[1:]} must be close to {predf}")


# Multivariate Gaussian distribution
print("\nMultivariate Gaussian distribution")
Q = np.arange(16).reshape(4, 4)
mean = np.array([1, 2, 3, 4])
data = np.random.randn(num_samples, 4) @ Q + mean

d0 = dist.MultivariateGaussianDistribution.from_samples(data, weights=weights)
d1 = MultivariateGaussianDistribution.from_samples(data, weights=weights)
print(f"{d0.mean} must be close to {mean}")
print(f"{d0.cov} must be close to {Q.T @ Q}")

check_equal([d0.mean.reshape(-1)])

# AR multvariate Gaussian distribution
print("\nAR multvariate Gaussian distribution")
