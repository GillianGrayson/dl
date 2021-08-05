import netket as nk
import numpy as np
from numpy import linalg as la
import pandas as pd
from netket.hilbert import Fock
from tqdm import tqdm

# Model params
N = 8
seed = 1
W = 20.0
U = 1.0
J = 1.0
gamma = 0.1

# Ansatz params
beta = 2
alpha = 2
n_samples = 2000
n_samples_diag = 2000
n_iter = 500

np.random.seed(seed)

# Uniformly distributed on-site energies
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
    A = (nk.operator.boson.create(hi, boson_id + 1) + nk.operator.boson.create(hi, boson_id)) * (nk.operator.boson.destroy(hi, boson_id + 1) - nk.operator.boson.destroy(hi, boson_id))
    j_ops.append(np.sqrt(gamma) * A)
ha += W * energies[N - 1] * nk.operator.boson.number(hi, N - 1)  # Don't forget last term

# Create the Liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

# Neural quantum state model: Positive-Definite Neural Density Matrix using the ansatz from Torlai and Melko
ndm = nk.models.NDM(
    alpha=alpha,
    beta=beta,
)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(lind.hilbert)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.vqs.MCMixedState(sa, ndm, n_samples=n_samples, n_samples_diag=n_samples_diag)
vs.init_parameters(nk.nn.initializers.normal(stddev=0.01))

# Driver
ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)

metrics_dict = {
    'iteration': np.linspace(1, n_iter+1, n_iter+1),
    'ldagl_mean': [],
    'ldagl_error_of_mean': [],
    'norm_rho_diff': [],
}

# Calculate exact rho
rho_exact = nk.exact.steady_state(lind, method="iterative", sparse=True, tol=1e-10)

for it in tqdm(range(n_iter)):
    out = ss.run(n_iter=1)
    metrics_dict['ldagl_mean'].append(ss.ldagl.mean)
    metrics_dict['ldagl_error_of_mean'].append(ss.ldagl.error_of_mean)
    rho_neural = np.array(ss.state.to_matrix())
    rho_diff = rho_exact - rho_neural
    rho_diff_conj = rho_exact - rho_neural.conjugate()
    metrics_dict['norm_rho_diff'].append(la.norm(rho_diff))

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_excel(f"metrics.xlsx", index=False)
