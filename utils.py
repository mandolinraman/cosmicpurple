"""Some utilties for computing softmax and sampling PMFs. 
"""


import numpy as np
from scipy.special import logsumexp


# Reducer for single destination
class Reducer:
    """_summary_"""

    def __init__(self, num: int):
        """_summary_

        Args:
            num (int): _description_
        """
        assert num >= 1

        self.pmf = np.ones(num) / num
        self.num = num
        self.winner = 0

    def reduce(self, metrics: np.ndarray) -> float:
        """_summary_

        Args:
            metrics (_type_): _description_

        Returns:
            _type_: _description_
        """

        assert len(metrics) == self.num

        self.winner = np.argmax(metrics)
        return metrics[self.winner]

    def soft_reduce(
        self, metrics: np.ndarray, compute_pmf: bool = False
    ) -> float:
        """_summary_

        Args:
            metrics (_type_): _description_
            compute_pmf (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        assert len(metrics) == self.num

        reduced = logsumexp(metrics)  # softmax(metrics)
        if compute_pmf:
            # first deal with corner cases:
            if reduced == np.inf:
                temp = metrics == np.inf
                self.pmf = temp / sum(temp)
            elif reduced == -np.inf:
                # all metrics must be -np.inf
                self.pmf.fill(1 / self.num)
            else:
                # normal case
                self.pmf = np.exp(metrics - reduced)

        return reduced

    def sample_pmf(self):
        """_summary_

        Args:
            metrics (_type_): _description_

        Returns:
            _type_: _description_
        """

        urand = np.random.rand()
        for index, prob in enumerate(self.pmf):
            urand -= prob
            if urand < 0:
                return index
        return self.num - 1


# Reducer for multiple destinations
class MultiReducer:
    """_summary_"""

    def __init__(self, num_buckets: int, buckets: np.ndarray):
        """Initialize a class instance

        Args:
            buckets (_type_): Bucket indices
        """
        assert min(buckets) >= 0 and max(buckets) < num_buckets

        self.num = len(buckets)
        self.buckets = buckets
        self.num_buckets = num_buckets
        self.winners = np.zeros(self.num_buckets)
        self.pmf = np.ones(self.num) / self.num

    def reduce(self, metrics: np.ndarray, reduced: np.ndarray):
        """_summary_

        Args:
            metrics (_type_): _description_
            reduced (_type_): _description_
        """
        assert len(metrics) == self.num
        assert len(reduced) == self.num_buckets

        reduced.fill(-np.inf)
        for index, (metric, bucket) in enumerate(zip(metrics, self.buckets)):
            if metric > reduced[bucket]:
                reduced[bucket] = metric
                self.winners[bucket] = index

    def soft_reduce(
        self, metrics: np.ndarray, reduced: np.ndarray, compute_pmf=False
    ):
        """_summary_

        Args:
            metrics (_type_): _description_
            reduced (_type_): _description_
            compute_pmf (bool, optional): _description_. Defaults to False.
        """

        assert len(metrics) == self.num
        assert len(reduced) == self.num_buckets

        reduced.fill(-np.inf)
        gamma = np.zeros_like(reduced)
        for metric, bucket in zip(metrics, self.buckets):
            # to prevent nan warnings when both terms are +/- np.inf:
            if reduced[bucket] == metric:
                delta = 0
            else:
                delta = reduced[bucket] - metric

            if delta < 0:
                gamma[bucket] = gamma[bucket] * np.exp(delta) + 1.0
                reduced[bucket] = metric
            else:
                gamma[bucket] += np.exp(-delta)

        reduced += np.log(gamma)  # softMax operation

        if compute_pmf:
            for index, (metric, bucket) in enumerate(
                zip(metrics, self.buckets)
            ):
                self.pmf[index] = (
                    1.0 / gamma[bucket]
                    if reduced[bucket] == metric
                    else np.exp(metric - reduced[bucket])
                )

    def sample_pmf(self, samples):
        """_summary_

        Args:
            samples (_type_): _description_

        Returns:
            _type_: _description_
        """

        assert len(samples) == self.num_buckets

        samples.fill(-1)
        urand = np.random.rand(self.num_buckets)
        for index, (prob, bucket) in enumerate(zip(self.pmf, self.buckets)):
            if samples[bucket] != -1:
                continue

            urand[bucket] -= prob
            if urand[bucket] < 0:
                samples[bucket] = index
