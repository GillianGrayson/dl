import netket as nk
import numpy as np


# Model params
L = 6
gp = 0.3
Vp = 2.0

# Ansatz params
beta = 2
alpha = 2
n_samples = 5000
n_samples_diag = 2000
n_iter = 1000

# Create graph
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# The hamiltonian
ha = nk.operator.LocalOperator(hi)

# List of dissipative jump operators
j_ops = []

for i in range(L):
    ha += (gp / 2.0) * nk.operator.spin.sigmax(hi, i)
    ha += (
        (Vp / 4.0)
        * nk.operator.spin.sigmaz(hi, i)
        * nk.operator.spin.sigmaz(hi, (i + 1) % L)
    )
    # sigma_{-} dissipation on every site
    j_ops.append(nk.operator.spin.sigmam(hi, i))

# Create the Liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

# Neural quantum state model: Positive-Definite Neural Density Matrix using the ansatz from Torlai and Melko
ndm = nk.models.NDM(
    alpha=alpha,
    beta=beta,
)

# Metropolis Local Sampling
# sa = nk.sampler.MetropolisLocal(lind.hilbert)
# sa = nk.sampler.MetropolisHamiltonian(lind.hilbert, hamiltonian=lind)
# sa = nk.sampler.MetropolisExchange(lind.hilbert, graph=g)

# Optimizer
#op = nk.optimizer.Sgd(0.01)
#sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.vqs.MCMixedState(sa, ndm, n_samples=n_samples, n_samples_diag=n_samples_diag)
vs.init_parameters(nk.nn.initializers.normal(stddev=0.01))

# # Driver
# ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)
#
# # Get batch og samples
# batch_of_samples = np.asarray(vs.samples.reshape((-1, vs.samples.shape[-1])))
#
# # Allowed states
# allowed_states = vs.hilbert.all_states()
#
# # Check samples
# num_non_exist = 0
# for st_id in range(0, batch_of_samples.shape[0]):
#     is_exist = np.equal(allowed_states, batch_of_samples[st_id, :]).all(1).any()
#     if not is_exist:
#         num_non_exist += 1
# print(f"Number of non-existing states in space: {num_non_exist} out of {batch_of_samples.shape[0]}")
