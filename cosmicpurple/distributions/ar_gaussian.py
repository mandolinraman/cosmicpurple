"Implementation of the Autoregressive Gaussian probability distribution"

import numpy as np
from ..utils import handle_nan, convert_to_array
from .distributions import Distribution


class ARGaussianDistribution(Distribution):
    """_summary_"""

    def __init__(self, mean: float, var: float, predictor: np.ndarray):
        """_summary_

        Args:
            pmf (nd.array): _description_
        """

        super().__init__()

        self.mean = mean
        self.var = var
        self.whitener = np.concatenate([[1.0], -predictor])
        self.d = len(self.whitener)
        self.summaries = np.zeros((self.d + 1, self.d + 1))

        self.half_inv_var = 0.5 / var
        self.log_const = 0.5 * np.log(2 * np.pi * var)

    @property
    def parameters(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return (self.mean, self.var, (-self.whitener[1:]).tolist())

    def log_probability(self, points: np.ndarray) -> float:
        """_summary_

        Args:
            i (int): _description_

        Returns:
            float: _description_
        """
        # points can be 1 dimensional (single data point)
        # or 2 dimensional (if we have a batch of points)
        error = np.dot(points, self.whitener) - self.mean
        log_prob = -self.log_const - self.half_inv_var * error**2
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

        col = np.zeros(self.d + 1)
        col[0] = 1
        vec = np.linalg.solve(self.summaries, col)
        new_whitener = vec[:-1] / vec[0]
        new_var = 1.0 / (self.summaries[-1, -1] * vec[0])
        new_mean = -vec[-1] / vec[0]

        self.whitener = inertia * self.whitener + (1 - inertia) * new_whitener
        self.mean = inertia * self.mean + (1 - inertia) * new_mean
        self.var = inertia * self.var + (1 - inertia) * new_var

        self.half_inv_var = 0.5 / self.var
        self.log_const = 0.5 * np.log(2 * np.pi * self.var)

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
        dist = cls.blank(points.shape[-1] - 1)
        dist.fit(points, weights)
        return dist

    @classmethod
    def blank(cls, predictor_len=1):
        """_summary_

        Returns:
            _type_: _description_
        """
        return cls(0, 1, np.zeros(predictor_len))
