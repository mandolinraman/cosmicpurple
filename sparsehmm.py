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
        iprob: np.ndarray = None,
        fprob: np.ndarray = None,
    ):
        """_summary_

        Args:
            adjList (_type_): _description_
            emission (_type_): _description_
            emission_map (_type_): _description_
            iprob (_type_, optional): _description_. Defaults to None.
            fprob (_type_, optional): _description_. Defaults to None.
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

        # validate iprob, fprob
        if iprob is None:
            iprob = np.ones(self.num_states) / self.num_states

        if fprob is None:
            fprob = np.ones(self.num_states)

        assert len(iprob) == self.num_states
        assert len(fprob) == self.num_states

        self.iprob = iprob
        self.fprob = fprob

    def _compute_log_emission_prob(self, obs, log_emission_prob):
        # compute all emission probabilities
        for j, emitter in enumerate(self.emitters):
            log_emission_prob[j] = emitter.log_probability(obs)

    def viterbi(self, obs: np.ndarray, temperature: float = 0):
        """_summary_

        Args:
            s (_type_): _description_
            sample (bool, optional): _description_. Defaults to False.
        """

        num_obs = len(obs)

        # max/softmax reducers:
        forward_reducer = MultiReducer(
            self.num_states, self.to_state, temperature
        )
        reducer = Reducer(self.num_states, temperature)

        # allocate space for trace back
        winning_edge = np.zeros((num_obs, self.num_states), int)
        metric = np.log(self.iprob)
        log_emission_prob = np.zeros(self.num_emitters)

        # start iterations
        for i in range(num_obs):
            # compute all accumulated path metrics:
            pathmetric = metric[self.from_state] + self.log_trans_prob
            if not np.isnan(obs[i]):
                self._compute_log_emission_prob(obs[i], log_emission_prob)
                pathmetric += log_emission_prob[self.emitter_map]
            # otherwise obs[i] == nan is treated as missing data
            # and we don't have an emission term.

            # if we want to sample the APP distribution, we apply
            # softmax to the converging metrics, rather than a max:
            if temperature > 0:
                # compute softmax and sample from softmax distribution
                forward_reducer.reduce(
                    pathmetric, output=metric, compute_pmf=True
                )
                forward_reducer.sample_softmax(winners=winning_edge[i])
            else:
                # compute simple max of converging metrics:
                forward_reducer.reduce(
                    pathmetric, output=metric, winners=winning_edge[i]
                )

        metric += np.log(self.fprob)

        # pick final state with least metric
        if temperature > 0:
            _ = reducer.reduce(metric, compute_pmf=True)
            ml_state = reducer.sample_softmax()
        else:
            _, ml_state = reducer.reduce(metric)

        # trace back to decode bits
        ml_seq = np.zeros(num_obs, int)
        for i in range(num_obs - 1, -1, -1):
            edge = winning_edge[i, ml_state]
            ml_state = self.from_state[edge]
            ml_seq[i] = edge

        return ml_seq

    def log_probability(self, obs: np.ndarray):
        """_summary_

        Args:
            obs (_type_): _description_

        Returns:
            _type_: _description_
        """

        num_obs = len(obs)

        # max/softmax reducers:
        forward_reducer = MultiReducer(self.num_states, self.to_state)
        reducer = Reducer(self.num_states)

        # allocate space
        metric = np.log(self.iprob)
        log_emission_prob = np.zeros(self.num_emitters)

        # start iterations
        for i in range(num_obs):
            # compute all accumulated path metrics:
            pathmetric = metric[self.from_state] + self.log_trans_prob
            if not np.isnan(obs[i]):
                self._compute_log_emission_prob(obs[i], log_emission_prob)
                pathmetric += log_emission_prob[self.emitter_map]
            # otherwise obs[i] == nan is treated as missing data
            # and we don't have an emission term.

            # apply softmax to the converging metrics:
            forward_reducer.reduce(pathmetric, output=metric)

        metric += np.log(self.fprob)
        return reducer.reduce(metric)

    def forward_backward(self, obs: np.ndarray):
        """_summary_

        Args:
            obs (np.ndarray): _description_

        Returns:
            _type_: _description_
        """

        num_obs = len(obs)

        # max/softmax reducers:
        forward_reducer = MultiReducer(self.num_states, self.to_state)
        backward_reducer = MultiReducer(self.num_states, self.from_state)
        reducer = Reducer(self.num_states)

        # compute and cache the emission probabilities
        log_emission_prob = np.zeros((num_obs, self.num_emitters))
        for i in range(num_obs):
            if not np.isnan(obs[i]):
                self._compute_log_emission_prob(
                    obs[i], log_emission_prob[i, self.emitter_map]
                )
            # otherwise obs[i] == nan is treated as missing data
            # and we don't have an emission term.

        # backward pass
        log_beta = np.zeros((num_obs, self.num_states))
        metric = np.log(self.fprob)
        for i in range(num_obs - 1, -1, -1):
            log_beta[i, :] = metric
            pathmetric = (
                metric[self.to_state]
                + self.log_trans_prob
                + log_emission_prob[i, self.emitter_map]
            )
            backward_reducer.reduce(pathmetric, output=metric)

        # forward pass
        # log_alpha = np.zeros((num_obs, self.num_states))
        log_app = np.zeros((num_obs, self.num_edges))
        metric = np.log(self.iprob)
        for i in range(num_obs):
            pathmetric = (
                metric[self.from_state]
                + self.log_trans_prob
                + log_emission_prob[i, self.emitter_map]
            )

            temp = pathmetric + log_beta[i, self.to_state]
            _ = reducer.reduce(temp, compute_pmf=True)
            log_app[:, i] = reducer.log_softmax_pmf
            forward_reducer.reduce(pathmetric, output=metric)

            # if we wanted to track the alphas too, we'd need this:
            # log_alpha[i, : ] = metric

        # entropy can be computed for almost no additional cost:
        metric += np.log(self.fprob)
        log_probability = reducer.reduce(metric)

        return log_app, log_probability
