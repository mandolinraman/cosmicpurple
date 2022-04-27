"""Some utilties for computing softmax and sampling softmax pmfs
in a numerically stable way.
"""

import numpy as np
from scipy.special import logsumexp
from numba import jit


@jit
def aggregate_simple(values, weights=None, small_weight=0.0):
    """_summary_

    Args:
        values (_type_): _description_
        num_buckets (_type_): _description_
        buckets (_type_): _description_

    Returns:
        _type_: _description_
    """
    result = np.zeros(values.shape[1:])
    if weights is None:
        for value in values:
            result += value
    else:
        for value, weight in zip(values, weights):
            if np.abs(weight) > small_weight:
                result += weight * value
    return result


@jit
def aggregate(values, num_buckets, buckets, weights=None, small_weight=0.0):
    """_summary_

    Args:
        values (_type_): _description_
        num_buckets (_type_): _description_
        buckets (_type_): _description_

    Returns:
        _type_: _description_
    """
    result = np.zeros((num_buckets,) + values.shape[1:])
    if weights is None:
        for value, bucket in zip(values, buckets):
            result[bucket] += value
    else:
        for value, bucket, weight in zip(values, buckets, weights):
            if np.abs(weight) > small_weight:
                result[bucket] += weight * value

    return result


@jit
def aggregate_max(
    metrics: np.ndarray,
    buckets: np.ndarray,
    output: np.ndarray,
    winners: np.ndarray,
):
    """_summary_

    Args:
        metrics (_type_): _description_
        reduced (_type_): _description_
    """
    output.fill(-np.inf)
    for index, (metric, bucket) in enumerate(zip(metrics, buckets)):
        if metric > output[bucket]:
            output[bucket] = metric
            winners[bucket] = index


@jit
def aggregate_softmax(
    metrics, buckets, output, compute_softmax, log_softmax_pmf
):
    """_summary_

    Args:
        metrics (_type_): _description_
        buckets (_type_): _description_
        output (_type_): _description_
        compute_softmax (_type_): _description_
        log_softmax_pmf (_type_): _description_
    """
    output.fill(-np.inf)
    gamma = np.zeros_like(output)
    for metric, bucket in zip(metrics, buckets):
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

    if compute_softmax:
        for index, (metric, bucket) in enumerate(zip(metrics, buckets)):
            log_softmax_pmf[index] = (
                -log_gamma[bucket]
                if output[bucket] == metric
                else metric - output[bucket]
            )


@jit
def sample_pmf(pmf):
    """_summary_

    Args:
        metrics (_type_): _description_

    Returns:
        _type_: _description_
    """

    urand = np.random.rand()
    winner = len(pmf) - 1
    for index, prob in enumerate(pmf):
        urand -= prob
        if urand < 0:
            winner = index
            break

    return winner


@jit
def sample_multi_pmf(pmf, buckets, winners):
    """_summary_

    Args:
        samples (_type_): _description_

    Returns:
        _type_: _description_
    """
    winners.fill(-1)
    urand = np.random.rand(len(winners))
    for index, (prob, bucket) in enumerate(zip(pmf, buckets)):
        if winners[bucket] != -1:
            continue

        urand[bucket] -= prob
        if urand[bucket] < 0:
            winners[bucket] = index


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

    def reduce(self, metrics):
        """_summary_

        Args:
            metric (_type_): _description_

        Returns:
            _type_: _description_
        """

        raise NotImplementedError


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

    def reduce(self, metrics: np.ndarray, compute_softmax=False):
        """_summary_

        Args:
            metrics (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        assert len(metrics) == self.num
        if self.temperature == 0.0:
            return self._hard_reduce(metrics)
        return self._soft_reduce(metrics, compute_softmax)

    def _hard_reduce(self, metrics: np.ndarray) -> float:
        """_summary_

        Args:
            metrics (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.winner = np.argmax(metrics)
        self.output = metrics[self.winner]
        return self.output, self.winner

    def _soft_reduce(
        self, metrics: np.ndarray, compute_softmax=False
    ) -> float:
        """_summary_

        Args:
            metrics (_type_): _description_
            compute_softmax (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if self.temperature != 1.0:
            metrics = metrics / self.temperature
        self.output = logsumexp(metrics)  # softmax(metrics)

        if compute_softmax:
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

        if self.temperature != 1.0:
            self.output *= self.temperature

        return self.output

    def sample_softmax(self):
        """_summary_

        Args:
            metrics (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.winner = sample_pmf(self.softmax_pmf)
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

    def reduce(
        self,
        metrics: np.ndarray,
        output=None,
        winners=None,
        compute_softmax=False,
    ):
        """_summary_

        Args:
            metrics (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        assert len(metrics) == self.num
        if output is None:
            output = self.output
        assert len(output) == self.num_buckets

        if self.temperature == 0:
            if winners is None:
                winners = self.winners
            assert len(winners) == self.num_buckets
            self._hard_reduce(metrics, output, winners)
        else:
            self._soft_reduce(metrics, output, compute_softmax)

    def _hard_reduce(
        self, metrics: np.ndarray, output: np.ndarray, winners: np.ndarray
    ):
        """_summary_

        Args:
            metrics (_type_): _description_
            reduced (_type_): _description_
        """
        aggregate_max(metrics, self.buckets, output, winners)

    def _soft_reduce(
        self, metrics: np.ndarray, output: np.ndarray, compute_softmax
    ):
        """_summary_

        Args:
            metrics (_type_): _description_
            reduced (_type_): _description_
            compute_softmax (bool, optional): _description_. Defaults to False.
        """
        if self.temperature != 1.0:
            metrics = metrics / self.temperature

        aggregate_softmax(
            metrics,
            self.buckets,
            output,
            compute_softmax,
            self.log_softmax_pmf,
        )

        if self.temperature != 1.0:
            output *= self.temperature

    def sample_softmax(self, winners=None):
        """_summary_

        Args:
            samples (_type_): _description_

        Returns:
            _type_: _description_
        """
        if winners is None:
            winners = self.winners
        assert len(winners) == self.num_buckets

        sample_multi_pmf(self.softmax_pmf, self.buckets, winners)
