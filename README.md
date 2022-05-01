# cosmicpurple

`cosmicpurple` is an sparse HMM library partly inspired by the [pomegranate](https://github.com/jmschrei/pomegranate) project. The goal of this project is to be modular and flexible with an API similar to that of `pomegranate`, but offer a few extra features including

- Edge-emitting models (in addition to state-emitting models).
- Streaming implementations using only the forward algorithm to compute posterior expectations and summarization of statistics in an extremely memory-efficient way..

The current implementation is in pure Python + numba for small speedups. 
