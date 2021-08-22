import netket as nk
import numpy as np
from numpy import linalg as la
import pandas as pd
import os
from tqdm import tqdm

# Model params
model = 'ising1d'
L = 6
gp = 0.3
Vp = 2.0

path = f"/media/sf_Work/dl/netket/{model}/L({L})_V({Vp})_g({gp})"
if not os.path.exists(f"{path}"):
    os.makedirs(f"{path}")

# Ansatz params
beta = 4
alpha = 4
n_samples = 4000
n_samples_diag = 1000
n_iter = 500

# Create graph
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# The hamiltonian
ha = nk.operator.LocalOperator(hi)

# List of dissipative jump operators
j_ops = []

# Observables
obs_sx = nk.operator.LocalOperator(hi)
obs_sy = nk.operator.LocalOperator(hi, dtype=complex)
obs_sz = nk.operator.LocalOperator(hi)

for i in range(L):
    ha += (gp / 2.0) * nk.operator.spin.sigmax(hi, i)
    ha += (
        (Vp / 4.0)
        * nk.operator.spin.sigmaz(hi, i)
        * nk.operator.spin.sigmaz(hi, (i + 1) % L)
    )
    # sigma_{-} dissipation on every site
    j_ops.append(nk.operator.spin.sigmam(hi, i))
    obs_sx += nk.operator.spin.sigmax(hi, i)
    obs_sy += nk.operator.spin.sigmay(hi, i)
    obs_sz += nk.operator.spin.sigmaz(hi, i)


# Create the liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

# Neural quantum state model: Positive-Definite Neural Density Matrix using the ansatz from Torlai and Melko
ma = nk.models.NDM(
    alpha=alpha,
    beta=beta,
)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(lind.hilbert)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

vs = nk.vqs.MCMixedState(sa, ma, n_samples=n_samples, n_samples_diag=n_samples_diag)
vs.init_parameters(nk.nn.initializers.normal(stddev=0.01))
ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)
obs = {"Sx": obs_sx, "Sy": obs_sy, "Sz": obs_sz}

rho_exact = nk.exact.steady_state(lind)
#rho_exact = nk.exact.steady_state(lind_for_exact, method="iterative", sparse=True, tol=1e-10)

metrics_dict = {
    'iteration': np.linspace(1, n_iter, n_iter),
    'ldagl_mean': [],
    'ldagl_error_of_mean': [],
    'norm_rho_diff_1': [],
    'norm_rho_diff_2': []
}

for it in tqdm(range(n_iter)):
    out = ss.run(n_iter=1)
    metrics_dict['ldagl_mean'].append(ss.ldagl.mean)
    metrics_dict['ldagl_error_of_mean'].append(ss.ldagl.error_of_mean)
    rho_neural = np.array(ss.state.to_matrix())
    rho_diff_1 = rho_exact - rho_neural
    rho_diff_2 = rho_exact - rho_neural.conjugate()
    metrics_dict['norm_rho_diff_1'].append(la.norm(rho_diff_1))
    metrics_dict['norm_rho_diff_2'].append(la.norm(rho_diff_2))

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_excel(f"{path}/metrics_size({alpha}_{beta})_samples({n_samples}_{n_samples_diag}).xlsx", index=False)

# ss.reset()
# out = ss.run(n_iter=n_iter, out="test", obs=obs)
# rho_neural = np.array(ss.state.to_matrix())
# rho_diff_1 = rho_exact - rho_neural
# rho_diff_2 = rho_exact - rho_neural.conjugate()
# print(f"norm_rho_diff_1: {la.norm(rho_diff_1)}")
# print(f"norm_rho_diff_2: {la.norm(rho_diff_2)}")