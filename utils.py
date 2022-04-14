"""Some utilties for computing softmax and sampling softmax pmfs
in a numerically stable way.
"""

import numpy as np
from scipy.special import logsumexp


class BaseReducer:
    """_summary_"""

    def __init__(self, num: int, temperature: float):
        """_summary_

        Args:
            num (int): _description_
        """
        assert num >= 1
        self.num = num
        self.log_softmax_pmf = np.full(self.num, -np.nan)

        assert temperature >= 0
        self.temperature = temperature

    @property
    def softmax_pmf(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return np.exp(self.log_softmax_pmf)

    def hard_reduce(self, metrics, **kwargs):
        """_summary_"""
        raise NotImplementedError

    def soft_reduce(self, metrics, **kwargs):
        """_summary_"""
        raise NotImplementedError

    def reduce(self, metrics, **kwargs):
        """_summary_

        Args:
            metric (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.temperature == 0:
            return self.hard_reduce(metrics, **kwargs)
        if self.temperature == 1:
            return self.soft_reduce(metrics, **kwargs)


# Reducer for single destination
class Reducer(BaseReducer):
    """_summary_"""

    def __init__(self, num: int, temperature: float = 1.0):
        """_summary_

        Args:
            num (int): _description_
        """
        super().__init__(num, temperature)
        self.winner = -1
        self.output = np.nan

    def hard_reduce(self, metrics: np.ndarray, **kwargs) -> float:
        """_summary_

        Args:
            metrics (_type_): _description_

        Returns:
            _type_: _description_
        """

        assert len(metrics) == self.num

        self.winner = np.argmax(metrics)
        self.output = metrics[self.winner]
        return self.output, self.winner

    def soft_reduce(self, metrics: np.ndarray, **kwargs) -> float:
        """_summary_

        Args:
            metrics (_type_): _description_
            compute_softmax (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        assert len(metrics) == self.num

        self.output = logsumexp(metrics)  # softmax(metrics)
        if kwargs.get("compute_softmax", False):
            # first deal with corner cases:
            if self.output == np.inf:
                temp = metrics == np.inf
                self.log_softmax_pmf.fill(-np.inf)
                self.log_softmax_pmf[temp] = -np.log(sum(temp))
            elif self.output == -np.inf:
                # all metrics must be -np.inf
                self.log_softmax_pmf.fill(-np.log(self.num))
            else:
                # normal case (all finite)
                self.log_softmax_pmf = metrics - self.output

        return self.output

    def sample_softmax(self):
        """_summary_

        Args:
            metrics (_type_): _description_

        Returns:
            _type_: _description_
        """

        urand = np.random.rand()
        self.winner = self.num - 1
        for index, prob in enumerate(self.softmax_pmf):
            urand -= prob
            if urand < 0:
                self.winner = index
                break

        return self.winner


# Reducer for multiple destinations
class MultiReducer(BaseReducer):
    """_summary_"""

    def __init__(
        self, num_buckets: int, buckets: np.ndarray, temperature: float = 1.0
    ):
        """Initialize a class instance

        Args:
            buckets (_type_): Bucket indices
        """
        super().__init__(len(buckets), temperature)

        assert min(buckets) >= 0 and max(buckets) < num_buckets
        self.buckets = buckets
        self.num_buckets = num_buckets
        self.output = np.full(num_buckets, np.nan)  # fall back for output
        self.winners = np.full(num_buckets, -1, int)  # fallback for winners

    def hard_reduce(self, metrics: np.ndarray, **kwargs):
        """_summary_

        Args:
            metrics (_type_): _description_
            reduced (_type_): _description_
        """
        output = kwargs.get("output", self.output)
        winners = kwargs.get("winners", self.winners)
        assert len(metrics) == self.num
        assert len(output) == self.num_buckets

        output.fill(-np.inf)
        for index, (metric, bucket) in enumerate(zip(metrics, self.buckets)):
            if metric > output[bucket]:
                output[bucket] = metric
                winners[bucket] = index

    def soft_reduce(self, metrics: np.ndarray, **kwargs):
        """_summary_

        Args:
            metrics (_type_): _description_
            reduced (_type_): _description_
            compute_softmax (bool, optional): _description_. Defaults to False.
        """
        output = kwargs.get("output", self.output)
        assert len(metrics) == self.num
        assert len(output) == self.num_buckets

        output.fill(-np.inf)
        gamma = np.zeros_like(output)
        for metric, bucket in zip(metrics, self.buckets):
            # to prevent nan warnings when both terms are +/- np.inf:
            if output[bucket] == metric:
                delta = 0
            else:
                delta = output[bucket] - metric

            if delta < 0:
                gamma[bucket] = gamma[bucket] * np.exp(delta) + 1.0
                output[bucket] = metric
            else:
                gamma[bucket] += np.exp(-delta)

        log_gamma = np.log(gamma)
        output += log_gamma  # softMax operation

        if kwargs.get("compute_softmax", False):
            for index, (metric, bucket) in enumerate(
                zip(metrics, self.buckets)
            ):
                self.log_softmax_pmf[index] = (
                    -log_gamma[bucket]
                    if output[bucket] == metric
                    else metric - output[bucket]
                )

    def sample_softmax(self, **kwargs):
        """_summary_

        Args:
            samples (_type_): _description_

        Returns:
            _type_: _description_
        """
        winners = kwargs.get("winners", self.winners)
        assert len(winners) == self.num_buckets

        winners.fill(-1)
        urand = np.random.rand(self.num_buckets)
        for index, (prob, bucket) in enumerate(
            zip(self.softmax_pmf, self.buckets)
        ):
            if winners[bucket] != -1:
                continue

            urand[bucket] -= prob
            if urand[bucket] < 0:
                winners[bucket] = index
