"""Some common distributions.
"""


import numpy as np


class Distribution:
    """_summary_"""

    def __init__(self):
        self.summaries = []

    @property
    def parameters(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return ()

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

    def process_summaries(self, process, other=None):
        """_summary_

        Args:
            process (_type_): _description_
        """
        # Apply the lambda function `process` to each tensor in summaries.
        # The default implementation works for
        # - a tensor
        # - a list of tensors
        # - a dictionary of tensors
        if other is None:
            if isinstance(self.summaries, np.ndarray):
                process(self.summaries)
            elif isinstance(self.summaries, list):
                for entry in self.summaries:
                    process(entry)
            elif isinstance(self.summaries, dict):
                for entry in self.summaries.values():
                    process(entry)
        else:
            if isinstance(self.summaries, np.ndarray):
                process(self.summaries, other.summaries)
            elif isinstance(self.summaries, list):
                for ours, theirs in zip(self.summaries, other.summaries):
                    process(ours, theirs)
            elif isinstance(self.summaries, dict):
                for key in self.summaries:
                    process(self.summaries[key], other.summaries[key])

    def clear_summaries(self):
        """_summary_"""
        # call fill(0) method on each tensor
        self.process_summaries(lambda tensor: tensor.fill(0))

    def scale_summaries(self, scale):
        """_summary_"""
        # __imul__ method does in place scaling of numpy arrays:
        self.process_summaries(lambda tensor: tensor.__imul__(scale))

    def steal_summaries_from(self, another, weight=1.0):
        """_summary_"""
        # __iadd__ does in place addition of numpy arrays
        self.process_summaries(
            lambda ours, theirs: ours.__iadd__(weight * theirs), another
        )

    def copy(self, steal_summaries=False):
        """_summary_

        Returns:
            _type_: _description_
        """
        clone = self.__class__(*self.parameters)
        if steal_summaries:
            clone.steal_summaries_from(self)

        return clone

    def summarize(self, points, weights):
        """_summary_

        Args:
            Self (_type_): _description_
            points (_type_): _description_
            weights (_type_): _description_
        """
        raise NotImplementedError

    def from_summaries(self, inertia=0.0):
        """_summary_

        Args:
            inertia (float, optional): _description_. Defaults to 0.0.
        """
        raise NotImplementedError

    def fit(self, points, weights=None, inertia=0.0):
        """_summary_

        Args:
            items (_type_): _description_
            weights (_type_, optional): _description_. Defaults to None.
            inertia (float, optional): _description_. Defaults to 0.0.
        """
        self.summarize(points, weights)
        self.from_summaries(inertia)
