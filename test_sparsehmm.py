"sparsehmm.py tests"
import numpy as np

import pomegranate as pom
import cosmicpurple as cp
import misc.nanoseek as ns


def signature(vec):
    """Reduces a vector to a single number (signature).

    Args:
        v (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.dot(vec.T.reshape(-1), np.sin(0.2 * np.arange(1, vec.size + 1)))


# load data
N = 1000
adjList = np.loadtxt("misc/wksp/adjList0.txt", int)
s = np.loadtxt("misc/wksp/S0.txt")[:N]
mu = np.loadtxt("misc/wksp/mu0.txt")
rv = np.loadtxt("misc/wksp/rv0.txt")
pc = np.loadtxt("misc/wksp/pc0.txt")
edges_hat = np.loadtxt("misc/wksp/edges_hat0.txt", int)[:N]

num_edges = len(adjList)
num_states = 1 + np.max(adjList)

emitters = [cp.GaussianDistribution(mu[t], rv[t]) for t in range(num_edges)]
hmm = cp.SparseHMM(adjList, pc, emitters)

# pomegranate
pdist = [
    pom.NormalDistribution(mu[t], np.sqrt(rv[t])) for t in range(num_edges)
]
pstates = [pom.State(pdist[t], f"state{t:02}") for t in range(num_edges)]

phmm = pom.HiddenMarkovModel()
phmm.add_states(*pstates)
for t in range(num_edges):
    phmm.add_transition(phmm.start, pstates[t], pc[t] / num_states)
    for u in range(num_edges):
        if adjList[t, 1] == adjList[u, 0]:
            phmm.add_transition(
                pstates[t], pstates[u], pc[u], group=f"group{u}"
            )

phmm.bake()

scale = 1.0

# viterbi
edges_hat_my = hmm.viterbi(s * scale)
edges_hat_pom = np.array(phmm.predict(s, "viterbi")[1:])
edges_hat_ns = ns.hmmViterbi(adjList, s, mu, rv, -np.log(pc))

err = np.sum(edges_hat != edges_hat_my)
print(f"Viterbi: number of errors = {err}")

h0_my = -hmm.log_probability(s * scale) / N / np.log(2)
h0_pom = -phmm.log_probability(s * scale) / N / np.log(2)
h0_ns = ns.hmmEntropy(adjList, s * scale, mu, rv, pc)
print(f"Entropy = {h0_my}, {h0_pom}, {h0_ns}")

logAPP_my, log_prob = hmm.forward_backward(s * scale)
trans_stats, logAPP_pom = phmm.forward_backward(s * scale)
logAPP_ns, h2_ns = ns.hmmFBA(adjList, s * scale, mu, rv, pc)

APP = np.exp(logAPP_my)

get_tensor = lambda i, obs: [
    np.eye(num_edges)[..., np.newaxis] * np.array([1, obs, obs**2])
]

momentQ_my1 = hmm.compute_expectation(
    s * scale, get_tensor, "forward_backward"
)[0]
momentQ_my2 = hmm.compute_expectation(s * scale, get_tensor, "forward")[0]
momentQ_ns, _ = ns.hmmMoments(adjList, s * scale, mu, rv, pc, s, 2)


# ##################
# # Classifier code
# ##################

# # data = sio.loadmat('misc/wksp.mat')
# # adjList = data['adjList'].astype(int)
# # x = data['x'].astype(int).squeeze()[:10]
# # s = data['S'].squeeze()[:10]
# # mu = data['mu']
# # rv = data['rv']
# # pc = data['pc'].squeeze()
# # px = data['px']

# adjList = np.loadtxt("misc/wksp/adjList.txt", int)
# x = np.loadtxt("misc/wksp/x.txt", int)[:N]
# s = np.loadtxt("misc/wksp/S.txt")[:N]
# mu = np.loadtxt("misc/wksp/mu.txt")
# rv = np.loadtxt("misc/wksp/rv.txt")
# pc = np.loadtxt("misc/wksp/pc.txt")
# px = np.loadtxt("misc/wksp/px.txt")

# small_probability = 0.0

# print("Case --")
# mlSeq1 = ns.hmmClassifierViterbi(adjList, x, s, px, mu, rv, -np.log(pc), False)
# mlSeq2 = ns.hmmClassifierViterbi(adjList, x, s, px, mu, rv, -np.log(pc), True)

# h0 = ns.hmmClassifierEntropy(adjList, x, s, px, mu, rv, pc)
# momentQ1, h1 = ns.hmmClassifierMoments(
#     adjList, x, s, px, mu, rv, pc, s, 2, small_probability, False
# )
# momentQ2, h2 = ns.hmmClassifierMoments(
#     adjList, x, s, px, mu, rv, pc, s, 2, small_probability, True
# )

# logAPP1, h3 = ns.hmmClassifierFBA(adjList, x, s, px, mu, rv, pc, False)
# logAPP2, h4 = ns.hmmClassifierFBA(adjList, x, s, px, mu, rv, pc, True)
# APP1 = np.exp(logAPP1)
# APP2 = np.exp(logAPP2)

# err1 = np.linalg.norm(mlSeq2 + x * 16 - mlSeq1)
# err2 = np.linalg.norm(momentQ1[:16] + momentQ1[16:] - momentQ2)
# err3 = np.linalg.norm(APP1[:16] + APP1[16:] - APP2)
# err4 = np.linalg.norm(np.sum(APP2, axis=0) - 1)
# err5 = np.linalg.norm(
#     momentQ1 - np.dot(APP1, s[:, np.newaxis] ** np.arange(3))
# )

# print("[%g, %g, %g, %g, %g]" % (err1, err2, err3, err4, err5))
# print(
#     "[%g, %g, %g]" % (signature(mlSeq1), signature(momentQ1), signature(APP1))
# )
# print("[%g, %g, %g, %g, %g]" % (h0, h1, h2, h3, h4))

# print("Case ?-")
# mlSeq1 = ns.hmmClassifierViterbi(
#     adjList, -np.ones(x.shape, int), s, px, mu, rv, -np.log(pc), False
# )
# mlSeq2 = ns.hmmClassifierViterbi(
#     adjList, -np.ones(x.shape, int), s, px, mu, rv, -np.log(pc), True
# )

# h0 = ns.hmmClassifierEntropy(
#     adjList, -np.ones(x.shape, int), s, px, mu, rv, pc
# )
# momentQ1, h1 = ns.hmmClassifierMoments(
#     adjList,
#     -np.ones(x.shape, int),
#     s,
#     px,
#     mu,
#     rv,
#     pc,
#     s,
#     2,
#     small_probability,
#     False,
# )
# momentQ2, h2 = ns.hmmClassifierMoments(
#     adjList,
#     -np.ones(x.shape, int),
#     s,
#     px,
#     mu,
#     rv,
#     pc,
#     s,
#     2,
#     small_probability,
#     True,
# )

# logAPP1, h3 = ns.hmmClassifierFBA(
#     adjList, -np.ones(x.shape, int), s, px, mu, rv, pc, False
# )
# logAPP2, h4 = ns.hmmClassifierFBA(
#     adjList, -np.ones(x.shape, int), s, px, mu, rv, pc, True
# )
# APP1 = np.exp(logAPP1)
# APP2 = np.exp(logAPP2)

# err1 = np.linalg.norm(mlSeq2 + x * 16 - mlSeq1)
# err2 = np.linalg.norm(momentQ1[:16] + momentQ1[16:] - momentQ2)
# err3 = np.linalg.norm(APP1[:16] + APP1[16:] - APP2)
# err4 = np.linalg.norm(np.sum(APP2, axis=0) - 1)
# err5 = np.linalg.norm(
#     momentQ1 - np.dot(APP1, s[:, np.newaxis] ** np.arange(3))
# )

# print("[%g, %g, %g, %g, %g]" % (err1, err2, err3, err4, err5))
# print(
#     "[%g, %g, %g]" % (signature(mlSeq1), signature(momentQ1), signature(APP1))
# )
# print("[%g, %g, %g, %g, %g]" % (h0, h1, h2, h3, h4))

# print("Case -?")
# mlSeq1 = ns.hmmClassifierViterbi(
#     adjList, x, s * np.nan, px, mu, rv, -np.log(pc), False
# )
# mlSeq2 = ns.hmmClassifierViterbi(
#     adjList, x, s * np.nan, px, mu, rv, -np.log(pc), True
# )

# h0 = ns.hmmClassifierEntropy(adjList, x, s * np.nan, px, mu, rv, pc)
# momentQ1, h1 = ns.hmmClassifierMoments(
#     adjList, x, s * np.nan, px, mu, rv, pc, s, 2, small_probability, False
# )
# momentQ2, h2 = ns.hmmClassifierMoments(
#     adjList, x, s * np.nan, px, mu, rv, pc, s, 2, small_probability, True
# )

# logAPP1, h3 = ns.hmmClassifierFBA(
#     adjList, x, s * np.nan, px, mu, rv, pc, False
# )
# logAPP2, h4 = ns.hmmClassifierFBA(adjList, x, s * np.nan, px, mu, rv, pc, True)
# APP1 = np.exp(logAPP1)
# APP2 = np.exp(logAPP2)

# err1 = np.linalg.norm(mlSeq2 + x * 16 - mlSeq1)
# err2 = np.linalg.norm(momentQ1[:16] + momentQ1[16:] - momentQ2)
# err3 = np.linalg.norm(APP1[:16] + APP1[16:] - APP2)
# err4 = np.linalg.norm(np.sum(APP2, axis=0) - 1)
# err5 = np.linalg.norm(
#     momentQ1 - np.dot(APP1, s[:, np.newaxis] ** np.arange(3))
# )

# print("[%g, %g, %g, %g, %g]" % (err1, err2, err3, err4, err5))
# print(
#     "[%g, %g, %g]" % (signature(mlSeq1), signature(momentQ1), signature(APP1))
# )
# print("[%g, %g, %g, %g, %g]" % (h0, h1, h2, h3, h4))

# print("Case ??")
# mlSeq1 = ns.hmmClassifierViterbi(
#     adjList, -np.ones(x.shape, int), s * np.nan, px, mu, rv, -np.log(pc), False
# )
# mlSeq2 = ns.hmmClassifierViterbi(
#     adjList, -np.ones(x.shape, int), s * np.nan, px, mu, rv, -np.log(pc), True
# )

# h0 = ns.hmmClassifierEntropy(
#     adjList, -np.ones(x.shape, int), s * np.nan, px, mu, rv, pc
# )
# momentQ1, h1 = ns.hmmClassifierMoments(
#     adjList,
#     -np.ones(x.shape, int),
#     s * np.nan,
#     px,
#     mu,
#     rv,
#     pc,
#     s,
#     2,
#     small_probability,
#     False,
# )
# momentQ2, h2 = ns.hmmClassifierMoments(
#     adjList,
#     -np.ones(x.shape, int),
#     s * np.nan,
#     px,
#     mu,
#     rv,
#     pc,
#     s,
#     2,
#     small_probability,
#     True,
# )

# logAPP1, h3 = ns.hmmClassifierFBA(
#     adjList, -np.ones(x.shape, int), s * np.nan, px, mu, rv, pc, False
# )
# logAPP2, h4 = ns.hmmClassifierFBA(
#     adjList, -np.ones(x.shape, int), s * np.nan, px, mu, rv, pc, True
# )
# APP1 = np.exp(logAPP1)
# APP2 = np.exp(logAPP2)

# err1 = np.linalg.norm(mlSeq2 + x * 16 - mlSeq1)
# err2 = np.linalg.norm(momentQ1[:16] + momentQ1[16:] - momentQ2)
# err3 = np.linalg.norm(APP1[:16] + APP1[16:] - APP2)
# err4 = np.linalg.norm(np.sum(APP2, axis=0) - 1)
# err5 = np.linalg.norm(
#     momentQ1 - np.dot(APP1, s[:, np.newaxis] ** np.arange(3))
# )

# print("[%g, %g, %g, %g, %g]" % (err1, err2, err3, err4, err5))
# print(
#     "[%g, %g, %g]" % (signature(mlSeq1), signature(momentQ1), signature(APP1))
# )
# print("[%g, %g, %g, %g, %g]" % (h0, h1, h2, h3, h4))
