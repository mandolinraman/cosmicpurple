"distributions.py tests"
import numpy as np
import pomegranate as pom
import distributions as di


def check_equal(first, second):
    if np.allclose(first, second):
        print("Passed!")
    else:
        print("Failed.")


num_samples = 100000
weights = np.random.rand(num_samples)

# Discrete distribution
print("\nDiscrete distribution")
data = np.floor(4 * np.random.rand(num_samples) ** 2).astype(int)

d0 = di.DiscreteDistribution.from_samples(data, 4, weights=weights)
d1 = pom.DiscreteDistribution.from_samples(data, weights=weights)
for i in range(4):
    print(f"{i}: ", d0.probability(i), d1.probability(i))

check_equal(
    [d0.probability(i) for i in range(4)],
    [d1.probability(i) for i in range(4)],
)

# Normal distribution
print("\nNormal distribution")
data = np.random.randn(num_samples) * 2 + 5
d0 = di.GaussianDistribution.from_samples(data, weights=weights)
d1 = pom.NormalDistribution.from_samples(data, weights=weights)
print("d0: [mean, std]", [d0.mean, np.sqrt(d0.var)])
print("d1: [mean, std]", d1.parameters)
check_equal([d0.mean, np.sqrt(d0.var)], d1.parameters)

# Multivariate Gaussian distribution
print("\nMultivariate Gaussian distribution")
Q = np.arange(16).reshape(4, 4)
mean = np.array([1, 2, 3, 4])
data = np.random.randn(num_samples, 4) @ Q + mean

d0 = di.MultivariateGaussianDistribution.from_samples(data, weights=weights)
d1 = pom.MultivariateGaussianDistribution.from_samples(data, weights=weights)
print(f"{d0.mean} must be close to {mean}")
print(f"{d0.cov}\n must be close to \n{Q.T @ Q}")


# AR Gaussian distribution
print("\nAR Gaussian distribution")
predf = np.array([0.5, -0.6, 0.9, 0.3])
data = np.random.randn(num_samples, 5)
bias, sigma = 3, 2
data[:, 0] = bias + data[:, 0] * sigma + data[:, 1:] @ predf
d0 = di.ARGaussianDistribution.from_samples(data, weights=weights)
d1 = di.MultivariateARGaussianDistribution.from_samples(
    data, 1, weights=weights
)


print(f"{d0.mean} must be close to {bias}")
print(f"{d0.var} must be close to {sigma**2}")
print(f"{d1.cov} must be close to {sigma**2}")
print(f"{-d0.whitener[1:]} must be close to {predf}")


# AR multivariate Gaussian distribution
print("\nAR multivariate Gaussian distribution")
predf = np.array([[0.5, -0.6], [0.1, 0.3]])
data = np.random.randn(num_samples, 4)
bias, sigma = np.array([1, 2]), 2
data[:, 0:2] = bias + data[:, 0:2] * sigma + data[:, 2:] @ predf
d1 = di.MultivariateARGaussianDistribution.from_samples(
    data, 2, weights=weights
)

print(f"{d1.mean} must be close to {bias}")
print(f"{d1.cov}\n must be close to\n {sigma**2 * np.eye(2)}")
print(f"{-d1.whitener[2:]}\n must be close to\n {predf}")
