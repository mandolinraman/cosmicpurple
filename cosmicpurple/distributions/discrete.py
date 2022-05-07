"Implementation of a discrete distribution"

import numpy as np
from ..utils import convert_to_array
from .distributions import Distribution


class DiscreteDistribution(Distribution):
    """_summary_"""

    def __init__(self, pmf):
        """_summary_

        Args:
            pmf (list, optional): _description_.
        """
        super().__init__()

        self.pmf = np.array(pmf)
        assert self.pmf.min() >= 0
        self.pmf /= self.pmf.sum()
        self.num_symbols = len(self.pmf)

        self.log_pmf = np.zeros(
            self.num_symbols + 1
        )  # one extra for missing data represented as "-1"
        self.log_pmf[: self.num_symbols] = np.log(self.pmf)
        self.d = 1
        self.summaries = np.zeros(self.num_symbols)

    @property
    def parameters(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return (self.pmf.tolist(),)

    def log_probability(self, points) -> float:
        """_summary_

        Args:
            i (int): _description_

        Returns:
            float: _description_
        """
        # points can be 0 dimensional (single data point)
        # or 1 dimensional (batch of points)
        if isinstance(points, np.ndarray):
            return self.log_pmf[points.astype(int)]
        return self.log_pmf[int(points)]
        # when a point is -1, it's interpereted as missing data
        # and we return self.log_pmf[-1] = 0

    def summarize(self, points: np.ndarray, weights=None):
        """_summary_

        Args:
            points (np.ndarray): _description_
            weights (_type_, optional): _description_. Defaults to None.
        """
        # points can be 0 dimensional (single data point)
        # or 1 dimensional (batch of points)
        points = convert_to_array(points, ndims=1, dtype=int)

        if weights is None:
            weights = np.ones(points.shape[0])
        else:
            weights = convert_to_array(weights, ndims=1)

        for i in range(self.num_symbols):
            self.summaries[i] += weights[points == i].sum()

    def from_summaries(self, inertia=0.0):
        """_summary_

        Args:
            inertia (float, optional): _description_. Defaults to 0.0.
        """

        new_pmf = self.summaries / self.summaries.sum()
        self.pmf = inertia * self.pmf + (1 - inertia) * new_pmf
        self.log_pmf[: self.num_symbols] = np.log(self.pmf)
        self.clear_summaries()

    @classmethod
    def from_samples(cls, points, num, weights=None):
        """_summary_

        Args:
            X (_type_): _description_
            weights (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        dist = cls.blank(num)
        dist.fit(points, weights)
        return dist

    @classmethod
    def blank(cls, num_symbols=1):
        """_summary_

        Returns:
            _type_: _description_
        """
        assert num_symbols >= 1
        return cls(np.full(num_symbols, 1.0 / num_symbols))
