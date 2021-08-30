import netket as nk
import numpy as np
import pandas as pd
from netket.hilbert import Fock
from tqdm import tqdm
import collections


# Model params
N = 8
seed = 42
W = 1.0
U = 1.0
J = 1.0
dt = 1
gamma = 0.1

# Ansatz params
beta = 2
alpha = 2
n_samples = 10000
n_iter = 10

np.random.seed(seed)
energies = np.random.uniform(-1.0, 1.0, N)

# Graph
g = nk.graph.Hypercube(N, n_dim=1, pbc=False)
# Hilbert space
hi = Fock(n_max=1, n_particles=N//2, N=g.n_nodes)

# The Hamiltonian
ha = nk.operator.LocalOperator(hi)
# List of dissipative jump operators
j_ops = []
for boson_id in range(N - 1):
    ha += W * energies[boson_id] * nk.operator.boson.number(hi, boson_id)
    ha += U * nk.operator.boson.number(hi, boson_id) * nk.operator.boson.number(hi, boson_id + 1)
    ha -= J * (nk.operator.boson.create(hi, boson_id) * nk.operator.boson.destroy(hi, boson_id + 1) + nk.operator.boson.create(hi, boson_id + 1) * nk.operator.boson.destroy(hi, boson_id))
    if dt == 1:
        A = (nk.operator.boson.create(hi, boson_id + 1) + nk.operator.boson.create(hi, boson_id)) * (nk.operator.boson.destroy(hi, boson_id + 1) - nk.operator.boson.destroy(hi, boson_id))
    elif dt == 0:
        A = nk.operator.boson.create(hi, boson_id) * nk.operator.boson.destroy(hi, boson_id)
    j_ops.append(np.sqrt(gamma) * A)
ha += W * energies[N - 1] * nk.operator.boson.number(hi, N - 1)
if dt == 0:
    A = nk.operator.boson.create(hi, N - 1) * nk.operator.boson.destroy(hi, N - 1)
    j_ops.append(np.sqrt(gamma) * A)

# Create the Liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

# Neural quantum state model: Positive-Definite Neural Density Matrix using the ansatz from Torlai and Melko
ndm = nk.models.NDM(
    alpha=alpha,
    beta=beta,
)

# Metropolis Local Sampling
graph = nk.graph.Hypercube(N, n_dim=1, pbc=False)
sa_graph = nk.graph.disjoint_union(graph, graph)
sa = nk.sampler.MetropolisExchange(lind.hilbert, graph=sa_graph)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.vqs.MCMixedState(sampler=sa, model=ndm, n_samples=n_samples)
vs.init_parameters(nk.nn.initializers.normal(stddev=0.01))

# Driver
ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)

# Allowed states
allowed_states = vs.hilbert.all_states()
states_dist = {}
for st in allowed_states:
    str_st = np.array2string(st, separator='')[1:-1].replace('.', '')
    states_dist[str_st] = 0

states_1d_str = []
for st in vs.hilbert_physical.all_states():
    str_st = np.array2string(st, separator='')[1:-1].replace('.', '')
    states_1d_str.append(str_st)

states_mtx = pd.DataFrame(data=np.zeros(shape=[len(states_1d_str), len(states_1d_str)]), index=states_1d_str, columns=states_1d_str)

for it in range(n_iter):

    # Get batch of samples
    batch_of_samples = np.asarray(vs.samples.reshape((-1, vs.samples.shape[-1])))

    # Check samples
    num_non_exist = 0
    for st_id in tqdm(range(0, batch_of_samples.shape[0])):
        is_exist = np.equal(allowed_states, batch_of_samples[st_id, :]).all(1).any()
        if not is_exist:
            num_non_exist += 1
        else:
            str_st = np.array2string(batch_of_samples[st_id, :], separator='')[1:-1].replace('.', '')
            states_dist[str_st] += 1
            states_mtx.loc[str_st[0:N], str_st[N::]] += 1
    print(f"Number of non-existing states in space: {num_non_exist} out of {batch_of_samples.shape[0]}")

    non_zero_states = np.count_nonzero(list(states_dist.values()))
    print(f"Number of non-zero states in allowed space: {non_zero_states} out of {len(allowed_states)}")

    vs.sample()
