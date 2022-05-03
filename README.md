# cosmicpurple

`cosmicpurple` is an sparse HMM library partly inspired by the [pomegranate](https://github.com/jmschrei/pomegranate) project. The goal of this project is to be modular and flexible with an API similar to that of `pomegranate`, but offer a few extra features including

- Edge-emitting models (in addition to state-emitting models).
- Support for sparse adjacency and multiple edges between states.
- Streaming implementations (using only the forward algorithm) to compute  expectations of quantities w.r.t. the posterior distribution, and summarization of statistics in an extremely memory-efficient way.

This is still a work in progress and the current implementation is in pure Python + numba for decent speedups. 
