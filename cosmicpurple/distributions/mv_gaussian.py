"Implementation of the Multivariate Gaussian probability distribution"

import numpy as np
from ..utils import handle_nan, convert_to_array
from .distributions import Distribution


class MultivariateGaussianDistribution(Distribution):
    """_summary_"""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """_summary_

        Args:
            pmf (nd.array): _description_
        """

        super().__init__()

        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.d = len(self.mean)
        self.summaries = np.zeros((self.d + 1, self.d + 1))

        # precompute
        self.half_inv_cov = 0.5 * np.linalg.inv(self.cov)
        self.log_const = 0.5 * self.d * np.log(2 * np.pi) + 0.5 * np.log(
            np.linalg.det(self.cov)
        )

    @property
    def parameters(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return (self.mean.tolist(), self.cov.tolist())

    def log_probability(self, points: np.ndarray) -> float:
        """_summary_

        Args:
            i (int): _description_

        Returns:
            float: _description_
        """
        # points can be 1 dimensional (single data point)
        # or 2 dimensional (if we have a batch of points)
        error = points - self.mean
        log_prob = -self.log_const - np.einsum(
            "...i, ij, ...j -> ...", error, self.half_inv_cov, error
        )
        # if some are nan, then it's treated as missing data
        # simply return 0
        return handle_nan(log_prob)

    def summarize(self, points: np.ndarray, weights=None):
        """_summary_

        Args:
            points (np.ndarray): _description_
            weights (_type_, optional): _description_. Defaults to None.
        """
        # points can be 1 dimensional (single data point)
        # or 2 dimensional (if we have a batch of points)
        points = convert_to_array(points, ndims=2)

        if weights is None:
            weights = np.ones(points.shape[0])
        else:
            weights = convert_to_array(weights, ndims=1)

        data = np.hstack([points, np.ones((points.shape[0], 1))])
        self.summaries += np.einsum("n, ni, nj -> ij", weights, data, data)

    def from_summaries(self, inertia=0.0):
        """_summary_

        Args:
            inertia (float, optional): _description_. Defaults to 0.0.
        """

        new_mean = self.summaries[-1, :-1] / self.summaries[-1, -1]
        new_cov = self.summaries[:-1, :-1] / self.summaries[
            -1, -1
        ] - np.einsum("i, j -> ij", new_mean, new_mean)

        self.mean = inertia * self.mean + (1 - inertia) * new_mean
        self.cov = inertia * self.cov + (1 - inertia) * new_cov

        self.half_inv_cov = 0.5 * np.linalg.inv(self.cov)
        self.log_const = 0.5 * self.d * np.log(2 * np.pi) + 0.5 * np.log(
            np.linalg.det(self.cov)
        )

        self.clear_summaries()

    @classmethod
    def from_samples(cls, points, weights=None):
        """_summary_

        Args:
            X (_type_): _description_
            weights (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        dist = cls.blank(points.shape[-1])
        dist.fit(points, weights)
        return dist

    @classmethod
    def blank(cls, ndim=2):
        """_summary_

        Returns:
            _type_: _description_
        """
        return cls(np.zeros(ndim), np.eye(ndim))
