"""Some common distributions.
"""


import numpy as np


class Multinomial:
    """_summary_"""

    def __init__(self, pmf: np.ndarray):
        """_summary_

        Args:
            pmf (nd.array): _description_
        """
        self.num = len(pmf)
        self.log_pmf = np.log(pmf / np.sum(pmf))

    def log_probability(self, i) -> float:
        """_summary_

        Args:
            i (int): _description_

        Returns:
            float: _description_
        """
        return self.log_pmf[int(i)]


class Normal:
    """_summary_"""

    def __init__(self, mean: float, var: float):
        """_summary_

        Args:
            pmf (nd.array): _description_
        """

        self.mean = mean
        self.var = var
        self.half_inv_var = 0.5 / var
        self.const = 0.5 * np.log(2 * np.pi * var)

    def log_probability(self, point: float) -> float:
        """_summary_

        Args:
            i (int): _description_

        Returns:
            float: _description_
        """

        return -self.const - self.half_inv_var * (point - self.mean) ** 2


class AutoRegressiveNormal:
    """_summary_"""

    def __init__(self, mean: float, var: float, predictor: np.ndarray):
        """_summary_

        Args:
            pmf (nd.array): _description_
        """

        self.mean = mean
        self.var = var
        self.whitener = np.hstack([1.0, -predictor])
        self.half_inv_var = 0.5 / var
        self.const = 0.5 * np.log(2 * np.pi * var)

    def log_probability(self, point: np.ndarray) -> float:
        """_summary_

        Args:
            i (int): _description_

        Returns:
            float: _description_
        """
        error = np.dot(self.whitener, point)
        return -self.const - self.half_inv_var * (error - self.mean) ** 2


class MultivariateGaussian:
    """_summary_"""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """_summary_

        Args:
            pmf (nd.array): _description_
        """

        self.mean = mean
        self.cov = cov
        self.half_inv_cov = 0.5 * np.linalg.inv(cov)
        self.const = 0.5 * np.log(2 * np.pi * np.linalg.det(cov))

    def log_probability(self, point: np.ndarray) -> float:
        """_summary_

        Args:
            i (int): _description_

        Returns:
            float: _description_
        """

        diff = point - self.mean
        return -self.const - np.einsum(
            "ij, jk, kl -> il", diff, self.half_inv_cov, diff
        )