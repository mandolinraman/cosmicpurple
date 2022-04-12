"""Library for Sparse HMMs
"""


import numpy as np
from utils import Reducer, MultiReducer


class SparseHMM:
    """_summary_"""

    def __init__(
        self,
        adj_list: np.ndarray,
        trans_prob: np.ndarray,
        emitters: np.ndarray,
        emitter_map: np.ndarray = None,
    ):
        """_summary_

        Args:
            adjList (_type_): _description_
            emission (_type_): _description_
            emission_map (_type_): _description_
        """

        # topology
        self.num_edges = len(adj_list)
        self.num_states = 1 + adj_list.max()
        self.from_state = adj_list[:, 0]
        self.to_state = adj_list[:, 1]

        # transition model
        assert len(trans_prob) == self.num_edges
        self.log_trans_prob = np.log(trans_prob)

        # emission model
        self.emitters = emitters  # emission distributions
        if emitter_map is None:
            emitter_map = self.to_state  # default is the "to" state

        assert len(emitter_map) == self.num_edges

        self.num_emitters = len(emitters)
        assert set(emitter_map) == set(range(self.num_emitters))
        self.emitter_map = emitter_map

    def viterbi(
        self,
        obs: np.ndarray,
        sample: bool = False,
        iprob: np.ndarray = None,
        fprob: np.ndarray = None,
    ):
        """_summary_

        Args:
            s (_type_): _description_
            sample (bool, optional): _description_. Defaults to False.
            iprob (_type_, optional): _description_. Defaults to None.
            fprob (_type_, optional): _description_. Defaults to None.
        """

        if iprob is None:
            iprob = np.ones(self.num_states) / self.num_states

        if fprob is None:
            fprob = np.ones(self.num_states)

        assert len(iprob) == self.num_states
        assert len(fprob) == self.num_states

        num_obs = len(obs)

        # max/softmax reducers:
        reducer = MultiReducer(self.num_states, self.to_state)
        final_reducer = Reducer(self.num_states)

        # allocate space for trace back
        winning_edge = np.zeros((num_obs, self.num_states), int)
        metric = np.log(iprob)
        log_emission_prob = np.zeros(self.num_emitters)

        # start iterations
        for i in range(num_obs):
            # compute all accumulated path metrics:
            pathmetric = metric[self.from_state] + self.log_trans_prob
            if not np.isnan(obs[i]):
                # compute all emission probabilities
                for j, emitter in enumerate(self.emitters):
                    log_emission_prob[j] = emitter.log_probability(obs[i])

                pathmetric += log_emission_prob[self.emitter_map]
            # otherwise obs[i] == nan, it's treated as missing data ans we
            # don't have an emission term

            # if we want to sample the APP distribution, we apply
            # softmax to the converging metrics, rather than a max:
            if sample:
                # compute softmax and sample from softmax distribution
                reducer.soft_reduce(pathmetric, metric, compute_pmf=True)
                reducer.sample_pmf(winning_edge[i])
            else:
                # compute simple max of converging metrics:
                reducer.reduce(pathmetric, metric)
                winning_edge[i] = reducer.winners

        metric += np.log(fprob)

        # pick final state with least metric
        if sample:
            _ = final_reducer.soft_reduce(metric, compute_pmf=True)
            ml_state = final_reducer.sample_pmf()
        else:
            _ = final_reducer.reduce(metric)
            ml_state = final_reducer.winner  # np.argmin(metric)

        # trace back to decode bits
        ml_seq = np.zeros(num_obs, int)
        for i in range(num_obs - 1, -1, -1):
            edge = winning_edge[i, ml_state]
            ml_state = self.from_state[edge]
            ml_seq[i] = edge

        return ml_seq

    def entropy(
        self,
        obs: np.ndarray,
        iprob: np.ndarray = None,
        fprob: np.ndarray = None,
    ):
        """_summary_

        Args:
            obs (_type_): _description_
            iprob (_type_, optional): _description_. Defaults to None.
            fprob (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        if iprob is None:
            iprob = np.ones(self.num_states) / self.num_states

        if fprob is None:
            fprob = np.ones(self.num_states)

        assert len(iprob) == self.num_states
        assert len(fprob) == self.num_states

        num_obs = len(obs)

        # max/softmax reducers:
        reducer = MultiReducer(self.num_states, self.to_state)
        final_reducer = Reducer(self.num_states)

        # allocate space
        metric = np.log(iprob)
        log_emission_prob = np.zeros(self.num_emitters)

        # start iterations
        for i in range(num_obs):
            # compute all accumulated path metrics:
            pathmetric = metric[self.from_state] + self.log_trans_prob
            if not np.isnan(obs[i]):
                # compute all emission probabilities
                for j, emitter in enumerate(self.emitters):
                    log_emission_prob[j] = emitter.log_probability(obs[i])

                pathmetric += log_emission_prob[self.emitter_map]
            # otherwise obs[i] == nan, it's treated as missing data ans we
            # don't have an emission term

            # apply softmax to the converging metrics:
            reducer.soft_reduce(pathmetric, metric, compute_pmf=True)

        metric += np.log(fprob)
        return -1 / num_obs / np.log(2.0) * final_reducer.soft_reduce(metric)
