"""Some common distributions.
"""


import numpy as np


def handle_nan(log_prob):
    """_summary_

    Args:
        log_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(log_prob, np.ndarray):
        log_prob[log_prob == np.nan] = 0
    elif np.isnan(log_prob):
        log_prob = 0.0

    return log_prob


class Distribution:
    """_summary_"""

    def __init__(self):
        pass

    def probability(self, points):
        """_summary_

        Args:
            points (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.exp(self.log_probability(points))

    def log_probability(self, points):
        """_summary_

        Args:
            points (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError


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
        if weights is None:
            weights = np.ones(points.shape[0])

        points = points.squeeze().astype(int)
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

    def clear_summaries(self):
        """_summary_"""
        self.summaries.fill(0)

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
        dist.summarize(points, weights)
        dist.from_summaries()
        return dist

    @classmethod
    def blank(cls, num_symbols=1):
        """_summary_

        Returns:
            _type_: _description_
        """
        assert num_symbols >= 1
        return cls(np.full(num_symbols, 1.0 / num_symbols))


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
        if weights is None:
            weights = np.ones(points.shape[0])

        points = points.squeeze()
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

    def clear_summaries(self):
        """_summary_"""
        self.summaries.fill(0)

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
        dist.summarize(points, weights)
        dist.from_summaries()
        return dist

    @classmethod
    def blank(cls):
        """_summary_

        Returns:
            _type_: _description_
        """
        return cls(0, 1)


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
        if weights is None:
            weights = np.ones(points.shape[0])

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

    def clear_summaries(self):
        """_summary_"""
        self.summaries.fill(0)

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
        dist.summarize(points, weights)
        dist.from_summaries()
        return dist

    @classmethod
    def blank(cls, ndim=2):
        """_summary_

        Returns:
            _type_: _description_
        """
        return cls(np.zeros(ndim), np.eye(ndim))


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
        if weights is None:
            weights = np.ones(points.shape[0])

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

    def clear_summaries(self):
        """_summary_"""
        self.summaries.fill(0)

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
        dist.summarize(points, weights)
        dist.from_summaries()
        return dist

    @classmethod
    def blank(cls, predictor_len=1):
        """_summary_

        Returns:
            _type_: _description_
        """
        return cls(0, 1, np.zeros(predictor_len))


class MultivariateARGaussianDistribution(Distribution):
    """_summary_"""

    def __init__(
        self, mean: np.ndarray, cov: np.ndarray, predictor: np.ndarray
    ):
        """_summary_

        Args:
            pmf (nd.array): _description_
        """

        super().__init__()

        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.n_dims = len(self.mean)
        self.whitener = np.concatenate([np.eye(self.n_dims), -predictor])
        self.d = self.whitener.shape[0]  # n_dims + predictor_len
        self.summaries = np.zeros((self.d + 1, self.d + 1))

        self.half_inv_cov = 0.5 * np.linalg.inv(self.cov)
        self.log_const = 0.5 * self.n_dims * np.log(2 * np.pi) + 0.5 * np.log(
            np.linalg.det(self.cov)
        )

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
        if weights is None:
            weights = np.ones(points.shape[0])

        data = np.hstack([points, np.ones((points.shape[0], 1))])
        self.summaries += np.einsum("n, ni, nj -> ij", weights, data, data)

    def from_summaries(self, inertia=0.0):
        """_summary_

        Args:
            inertia (float, optional): _description_. Defaults to 0.0.
        """
        predictor = np.linalg.solve(
            self.summaries[self.n_dims :, self.n_dims :],
            self.summaries[self.n_dims :, : self.n_dims],
        )
        new_whitener = np.concatenate(
            [np.eye(self.n_dims), -predictor[:-1, :]]
        )
        new_mean = predictor[-1, :]
        new_cov = (
            self.summaries[: self.n_dims, : self.n_dims]
            - np.dot(self.summaries[: self.n_dims, self.n_dims :], predictor)
        ) / self.summaries[-1, -1]

        self.whitener = inertia * self.whitener + (1 - inertia) * new_whitener
        self.mean = inertia * self.mean + (1 - inertia) * new_mean
        self.cov = inertia * self.cov + (1 - inertia) * new_cov

        self.half_inv_cov = 0.5 * np.linalg.inv(self.cov)
        self.log_const = 0.5 * self.n_dims * np.log(2 * np.pi) + 0.5 * np.log(
            np.linalg.det(self.cov)
        )

        self.clear_summaries()

    def clear_summaries(self):
        """_summary_"""
        self.summaries.fill(0)

    @classmethod
    def from_samples(cls, points, n_dims, weights=None):
        """_summary_

        Args:
            X (_type_): _description_
            weights (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        dist = cls.blank(n_dims, points.shape[-1] - n_dims)
        dist.summarize(points, weights)
        dist.from_summaries()
        return dist

    @classmethod
    def blank(cls, n_dims=1, predictor_len=1):
        """_summary_

        Returns:
            _type_: _description_
        """
        return cls(
            np.zeros(n_dims), np.eye(n_dims), np.zeros((predictor_len, n_dims))
        )
