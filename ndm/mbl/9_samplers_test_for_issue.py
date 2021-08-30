import netket as nk
import numpy as np
from netket.hilbert import Fock
from tqdm import tqdm


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

# Hilbert space
hi = Fock(n_max=1, n_particles=N//2, N=N)

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

# Metropolis Sampling
graph = nk.graph.Hypercube(10, n_dim=1, pbc=False)
sa = nk.sampler.MetropolisExchange(lind.hilbert, graph=graph)

# Variational state
vs = nk.vqs.MCMixedState(sampler=sa, model=ndm, n_samples=n_samples)
vs.init_parameters(nk.nn.initializers.normal(stddev=0.01))

# Allowed states
allowed_states = vs.hilbert.all_states()

# Allowed states statistics
states_dist = {}
for st in allowed_states:
    str_st = np.array2string(st, separator='')[1:-1].replace('.', '')
    states_dist[str_st] = 0

for it in range(n_iter):

    # Get batch of samples
    batch_of_samples = np.asarray(vs.samples.reshape((-1, vs.samples.shape[-1])))

    # Check samples
    num_non_exist = 0
    for st_id in tqdm(range(0, batch_of_samples.shape[0])):
        str_st = np.array2string(batch_of_samples[st_id, :], separator='')[1:-1].replace('.', '')
        if str_st not in states_dist:
            num_non_exist += 1
        else:
            str_st = np.array2string(batch_of_samples[st_id, :], separator='')[1:-1].replace('.', '')
            states_dist[str_st] += 1

    print(f"Number of non-existing states in space: {num_non_exist} out of {batch_of_samples.shape[0]}")

    non_zero_states = np.count_nonzero(list(states_dist.values()))
    print(f"Number of non-zero states in allowed space: {non_zero_states} out of {len(allowed_states)}")

    vs.sample()
