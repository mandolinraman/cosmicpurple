"Implementation of the Gaussian probability distribution"

import numpy as np
from ..utils import handle_nan, convert_to_array
from .distributions import Distribution


class GaussianDistribution(Distribution):
    """_summary_"""

    def __init__(self, mean: float, var: float):
        """_summary_

        Args:
            mean (float): _description_
            var (float): _description_
        """
        super().__init__()

        self.mean = mean
        self.var = var
        # self.parameters = (self.mean, self.var)
        self.d = 1
        self.summaries = np.zeros(3)

        # precompute
        self.half_inv_var = 0.5 / self.var
        self.log_const = 0.5 * np.log(2 * np.pi * self.var)

    @property
    def parameters(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return (self.mean, self.var)

    def log_probability(self, points: float) -> float:
        """_summary_

        Args:
            points (int): _description_

        Returns:
            float: _description_
        """
        # points can be 0 dimensional (single data point)
        # or 1 dimensional (batch of points)
        log_prob = (
            -self.log_const - self.half_inv_var * (points - self.mean) ** 2
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
        points = convert_to_array(points, ndims=1)

        if weights is None:
            weights = np.ones(points.shape[0])
        else:
            weights = convert_to_array(weights, ndims=1)

        self.summaries[0] += weights.sum()
        self.summaries[1] += weights.dot(points)
        self.summaries[2] += weights.dot(points**2)

    def from_summaries(self, inertia=0.0):
        """_summary_

        Args:
            inertia (float, optional): _description_. Defaults to 0.0.
        """
        new_mean = self.summaries[1] / self.summaries[0]
        new_var = self.summaries[2] / self.summaries[0] - new_mean**2

        self.mean = inertia * self.mean + (1 - inertia) * new_mean
        self.var = inertia * self.var + (1 - inertia) * new_var

        # self.parameters = (self.mean, self.var)
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
        dist = cls.blank()
        dist.fit(points, weights)
        return dist

    @classmethod
    def blank(cls):
        """_summary_

        Returns:
            _type_: _description_
        """
        return cls(0, 1)
