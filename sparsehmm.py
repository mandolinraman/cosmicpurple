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
            emitter_map = np.arange(
                self.num_edges
            )  # default is the edge index

        assert len(emitter_map) == self.num_edges

        self.num_emitters = len(emitters)
        assert set(emitter_map) == set(range(self.num_emitters))
        self.edge_to_emitter = emitter_map

        # validate iprob, fprob
        if iprob is None:
            iprob = np.ones(self.num_states) / self.num_states

        if fprob is None:
            fprob = np.ones(self.num_states)

        assert len(iprob) == self.num_states
        assert len(fprob) == self.num_states

        self.iprob = iprob
        self.fprob = fprob

        # summaries for computing edge transition probabilities
        self.summaries = np.zeros(self.num_edges)

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
        state_reducer = Reducer(self.num_states, temperature)

        # allocate space for trace back
        winning_edge = np.zeros((num_obs, self.num_states), int)
        metric = np.log(self.iprob)
        log_emission_prob = np.zeros(self.num_emitters)

        # start iterations
        for i in range(num_obs):
            self._compute_log_emission_prob(obs[i], log_emission_prob)
            # compute all accumulated path metrics:
            pathmetric = (
                metric[self.from_state]
                + self.log_trans_prob
                + log_emission_prob[self.edge_to_emitter]
            )

            # if we want to sample the APP distribution, we apply
            # softmax to the converging metrics, rather than a max:
            if temperature > 0:
                # compute softmax and sample from softmax distribution
                forward_reducer.reduce(
                    pathmetric, output=metric, compute_softmax=True
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
            _ = state_reducer.reduce(metric, compute_softmax=True)
            ml_state = state_reducer.sample_softmax()
        else:
            _, ml_state = state_reducer.reduce(metric)

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
        state_reducer = Reducer(self.num_states)

        # allocate space
        metric = np.log(self.iprob)
        log_emission_prob = np.zeros(self.num_emitters)

        # start iterations
        for i in range(num_obs):
            self._compute_log_emission_prob(obs[i], log_emission_prob)
            # compute all accumulated forward path metrics:
            pathmetric = (
                metric[self.from_state]
                + self.log_trans_prob
                + log_emission_prob[self.edge_to_emitter]
            )

            # apply softmax to the converging metrics:
            forward_reducer.reduce(pathmetric, output=metric)

        metric += np.log(self.fprob)
        return state_reducer.reduce(metric)

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
        state_reducer = Reducer(self.num_states)
        edge_reducer = Reducer(self.num_edges)

        # precompute and cache the emission probabilities
        log_emission_prob = np.zeros((num_obs, self.num_emitters))
        for i in range(num_obs):
            self._compute_log_emission_prob(obs[i], log_emission_prob[i])

        # forward pass
        # log_alpha = np.zeros((num_obs, self.num_states))
        log_app = np.zeros((num_obs, self.num_edges))
        metric = np.log(self.iprob)
        for i in range(num_obs):
            # compute all accumulated forward path metrics:
            pathmetric = (
                metric[self.from_state]
                + self.log_trans_prob
                + log_emission_prob[i, self.edge_to_emitter]
            )

            # cache the forward pathmetrics in log-app:
            log_app[i] = pathmetric
            forward_reducer.reduce(pathmetric, output=metric)

            # uncomment if you want to track the forward coefficients:
            # log_alpha[i] = metric

        # log probability can be computed for almost no additional cost:
        metric += np.log(self.fprob)
        log_probability = state_reducer.reduce(metric)

        # backward pass
        metric = np.log(self.fprob)
        for i in range(num_obs - 1, -1, -1):
            to_state_metric = metric[self.to_state]

            # compute a posteriori probabilies (APP) for edge:
            edge_reducer.reduce(
                log_app[i] + to_state_metric, compute_softmax=True
            )
            # overwrite log_app[i] with final APP for edge:
            log_app[i] = edge_reducer.log_softmax_pmf

            # compute all accumulated backward path metrics:
            pathmetric = (
                to_state_metric
                + self.log_trans_prob
                + log_emission_prob[i, self.edge_to_emitter]
            )
            backward_reducer.reduce(pathmetric, output=metric)

        return log_app, log_probability

    def summarize(self, obs: np.ndarray, algorithm, temperature=0.0):
        """_summary_

        Args:
            obs (np.ndarray): _description_
        """

        if algorithm == "forward_backward":
            self._forward_backward_summarize(obs)
        elif algorithm == "forward_summarize":
            self._forward_summarize(obs)
        elif algorithm == "viterbi":
            pass

    def _forward_backward_summarize(self, obs: np.ndarray):
        """_summary_

        Args:
            obs (_type_): _description_

        Returns:
            _type_: _description_
        """

        log_app, _ = self.forward_backward(obs)
        app = np.exp(log_app)
        # log_edge_count = logsumexp(log_app, axis=0, compute_softmax=True)
        edge_count = app.sum(axis=0)

        # summarize for transition model
        self.summaries[0] += edge_count

        # let each emitter summarize
        for edge, emitter_index in enumerate(self.edge_to_emitter):
            self.emitters[emitter_index].summarize(obs, app[:, edge])

    def _forward_summarize(self, obs: np.ndarray):
        """_summary_

        Args:
            obs (np.ndarray): _description_
        """

    def from_summaries(self, inertia=0.0):
        """_summary_

        Args:
            inertia (float, optional): _description_. Defaults to 0.0.
        """

        # TODO: estimate transition model

        self.clear_summaries()

        # compute new model from summaries for each *unique* emitter
        # and then clear its summary
        for emitter in self.emitters:
            emitter.from_summaries(inertia)
            emitter.clear_summaries()

    def clear_summaries(self):
        """_summary_"""
        self.summaries.fill(0)

    def compute_expectation(
        self, obs: np.ndarray, get_tensor, algorithm, small_probability=0.0
    ):
        """_summary_

        Args:
            obs (np.ndarray): _description_
            get_tensor (_type_, optional): _description_. Defaults to None.
            small_probability (float, optional): _description_. Defaults to 0.0.
        """
        if algorithm == "forward":
            return self._forward_expectation(
                obs, get_tensor, small_probability
            )

        if algorithm == "forward_backward":
            pass
        else:
            raise ValueError("Invalid algorithm")

    def _forward_backward_expectation(
        self, obs: np.ndarray, get_tensor=None, small_probability=0.0
    ):
        """_summary_

        Args:
            obs (np.ndarray): _description_
            get_tensor (_type_, optional): _description_. Defaults to None.
            small_probability (float, optional): _description_. Defaults to 0.0.
        """
        num_obs = len(obs)

        log_app, _ = self.forward_backward(obs)
        app = np.exp(log_app)

        for i in range(num_obs):
            tensors = get_tensor(i, obs[i])
            if i == 0:
                # first time setup:
                expectation = [
                    np.zeros(tensor.shape[1:]) for tensor in tensors
                ]

            for exp, tensor in zip(expectation, tensors):
                for edge, prob in enumerate(app[i]):
                    if prob > small_probability:
                        exp += prob * tensor[edge]
                # it's same as this:
                # exp += np.einsum("e, e... -> ...", app[i], tensor)

        return expectation

    def _forward_expectation(
        self, obs: np.ndarray, get_tensor=None, small_probability=0.0
    ):
        """_summary_

        Args:
            obs (_type_): _description_

        Returns:
            _type_: _description_
        """

        num_obs = len(obs)

        # max/softmax reducers:
        forward_reducer = MultiReducer(self.num_states, self.to_state)
        state_reducer = Reducer(self.num_states)

        # allocate space
        metric = np.log(self.iprob)
        log_emission_prob = np.zeros(self.num_emitters)

        # start iterations
        for i in range(num_obs):
            self._compute_log_emission_prob(obs[i], log_emission_prob)
            # compute all accumulated forward path metrics:
            pathmetric = (
                metric[self.from_state]
                + self.log_trans_prob
                + log_emission_prob[self.edge_to_emitter]
            )

            # apply softmax to the converging metrics:
            forward_reducer.reduce(
                pathmetric, output=metric, compute_softmax=True
            )
            pmf = forward_reducer.softmax_pmf

            # Get a list of tensors that need to be averaged w.r.t.
            # the posterior probability. The 1st dimension of each
            # tensor is the edge index:
            #   tensors[i][e, ...] = the i-th tensor on edge e

            tensors = get_tensor(i, obs[i])
            if i == 0:
                # first time setup:
                expectation = [
                    np.zeros((self.num_states,) + tensor.shape[1:])
                    for tensor in tensors
                ]
                new_expectation = [
                    np.zero((self.num_states,) + tensor.shape[1:])
                    for tensor in tensors
                ]
            else:
                for tensor in new_expectation:
                    tensor.fill(0)

            for exp, new_exp, tensor in zip(
                expectation, new_expectation, tensors
            ):
                for edge, prob in enumerate(pmf):
                    if prob > small_probability:
                        new_exp[self.to_state[edge]] += prob * (
                            exp[self.from_state[edge]] + tensor[edge]
                        )

            # swap the two
            expectation, new_expectation = new_expectation, expectation

        metric += np.log(self.fprob)
        return state_reducer.reduce(metric)
