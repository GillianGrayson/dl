import netket as nk
import numpy as np
from numpy import linalg as la
import pandas as pd
import os
import re
from netket.hilbert import Fock

# Model params
model = 'mbl'
N = 8

W = 20.0
U = 1.0
J = 1.0

gamma = 0.1

cpp_seed = 10

# Hilbert space
hi = Fock(n_max=1, n_particles=N//2, N=N)

cpp_path = f"/media/sf_Work/dl/netket/{model}/test/cpp"
save_path = f"/media/sf_Work/dl/netket/{model}/N({N})_H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({gamma:0.4f}))"
if not os.path.exists(f"{save_path}"):
    os.makedirs(f"{save_path}")

energies = np.loadtxt(
    f"{cpp_path}/energies_ns({N})_seed({cpp_seed})_diss(1_0.0000_{gamma:0.4f})_prm({W:0.4f}_{U:0.4f}_{J:0.4f}).txt",
    delimiter='\n')
with open(f"{cpp_path}/hamiltonian_mtx_ns({N})_seed({cpp_seed})_diss(1_0.0000_{gamma:0.4f})_prm({W:0.4f}_{U:0.4f}_{J:0.4f}).txt") as f:
    content = f.read().splitlines()
ha_cpp = np.zeros(shape=(hi.n_states, hi.n_states), dtype='complex128')
for row in content:
    regex = '^(\d+)\t(\d+)\t\(([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?),([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?)\)$'
    strings = re.search(regex, row).groups()
    (i, j, re, im) = [t(s) for t, s in zip((int, int, float, float), )]
    ha_cpp[i, j] = re + 1j * im

# Hilbert space
hi = Fock(n_max=1, n_particles=N//2, N=N)

# The Hamiltonian
ha = nk.operator.LocalOperator(hi)

# List of dissipative jump operators
j_ops = []

for i in range(N - 1):
    ha += W * energies[i] * nk.operator.boson.number(hi, i)
    ha += U * nk.operator.boson.number(hi, i) * nk.operator.boson.number(hi, i + 1)
    ha -= J * (nk.operator.boson.create(hi, i) * nk.operator.boson.destroy(hi, i + 1) + nk.operator.boson.create(hi, i + 1) * nk.operator.boson.destroy(hi, i))
    A = np.sqrt(gamma) * (nk.operator.boson.create(hi, i) + nk.operator.boson.create(hi, i + 1)) * (nk.operator.boson.destroy(hi, i) + nk.operator.boson.destroy(hi, i + 1))
    j_ops.append(A)
ha += W * energies[N - 1] * nk.operator.boson.number(hi, N - 1)

ha_dense = ha.to_dense()



# Create the liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)
rho_it = nk.exact.steady_state(
    lind, method="iterative", sparse=True, tol=1e-8
)
rho_ed = nk.exact.steady_state(lind)

ololo = 1

