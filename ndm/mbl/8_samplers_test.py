import netket as nk
import numpy as np
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
n_samples = 5000
n_samples_diag = 2000
n_iter = 1000

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
# sa = nk.sampler.MetropolisLocal(lind.hilbert)
# sa = nk.sampler.MetropolisExchange(lind.hilbert, graph=g)
sa = nk.sampler.MetropolisHamiltonian(lind.hilbert, hamiltonian=lind)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
#vs = nk.vqs.MCMixedState(sampler=sa, model=ndm, n_samples=n_samples, n_samples_diag=n_samples_diag)
vs = nk.vqs.MCMixedState(sampler=sa, model=ndm, n_samples=n_samples, n_samples_diag=0)
vs.init_parameters(nk.nn.initializers.normal(stddev=0.01))

# Driver
ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)

# Get batch of samples
batch_of_samples = np.asarray(vs.samples.reshape((-1, vs.samples.shape[-1])))

# Allowed states
allowed_states = vs.hilbert.all_states()

# Check samples
num_non_exist = 0
for st_id in range(0, batch_of_samples.shape[0]):
    is_exist = np.equal(allowed_states, batch_of_samples[st_id, :]).all(1).any()
    if not is_exist:
        num_non_exist += 1
print(f"Number of non-existing states in space: {num_non_exist} out of {batch_of_samples.shape[0]}")
