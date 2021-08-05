import netket as nk
import numpy as np
from numpy import linalg as la
import pandas as pd
import os
import re
from netket.hilbert import Fock

# Model params
model = 'mbl'
N = 6

W = 20.0
U = 1.0
J = 1.0

dt = 1
gamma = 0.1

cpp_seed = 10

# Hilbert space
hi = Fock(n_max=1, n_particles=N//2, N=N)

cpp_path = f"/media/sf_Work/dl/netket/{model}/test/cpp"
save_path = f"/media/sf_Work/dl/netket/{model}/N({N})_H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({dt}_{gamma:0.4f})"
if not os.path.exists(f"{save_path}"):
    os.makedirs(f"{save_path}")

energies = np.loadtxt(
    f"{cpp_path}/energies_ns({N})_seed({cpp_seed})_diss({dt}_0.0000_{gamma:0.4f})_prm({W:0.4f}_{U:0.4f}_{J:0.4f}).txt",
    delimiter='\n')
with open(f"{cpp_path}/hamiltonian_mtx_ns({N})_seed({cpp_seed})_diss({dt}_0.0000_{gamma:0.4f})_prm({W:0.4f}_{U:0.4f}_{J:0.4f}).txt") as f:
    content = f.read().splitlines()
ha_cpp = np.zeros(shape=(hi.n_states, hi.n_states), dtype='float64')
for row in content:
    regex = '^(\d+)\t(\d+)\t\(([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?),([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?)\)$'
    strings = re.search(regex, row).groups()
    (boson_id, j, real, image) = [t(s) for t, s in zip((int, int, float, float), strings)]
    if abs(image) > 1e-10:
        print('Non-zero imag part in cpp Hamiltonian')
    ha_cpp[boson_id, j] = real

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

for A_id, A in enumerate(j_ops):
    A_dense = A.to_dense()
    with open(f"{cpp_path}/diss_{len(j_ops) - 1 - A_id}_mtx_ns({N})_seed({cpp_seed})_diss({dt}_0.0000_{gamma:0.4f})_prm({W:0.4f}_{U:0.4f}_{J:0.4f}).txt") as f:
        content = f.read().splitlines()
    A_cpp = np.zeros(shape=(hi.n_states, hi.n_states), dtype='float64')
    for row in content:
        regex = '^(\d+)\t(\d+)\t\(([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?),([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?)\)$'
        strings = re.search(regex, row).groups()
        (row_id, col_id, real, image) = [t(s) for t, s in zip((int, int, float, float), strings)]
        if abs(image) > 1e-10:
            print(f'Non-zero imag part in cpp A {A_id}')
        A_cpp[row_id, col_id] = np.sqrt(gamma) * real
    A_diff = la.norm(A_cpp - A_dense)
    print(f'A_diff {A_id}: {A_diff}')

ha_dense = ha.to_dense()
ha_diff = la.norm(ha_cpp - ha_dense)
print(f'ha_diff: {ha_diff}')

# Create the Liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)
lind_dense = lind.to_dense()
with open(f"{cpp_path}/lindbladian_mtx_ns({N})_seed({cpp_seed})_diss({dt}_0.0000_{gamma:0.4f})_prm({W:0.4f}_{U:0.4f}_{J:0.4f}).txt") as f:
    content = f.read().splitlines()
lind_cpp = np.zeros(shape=(hi.n_states * hi.n_states, hi.n_states * hi.n_states), dtype='complex128')
for row in content:
    regex = '^(\d+)\t(\d+)\t\(([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?),([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?)\)$'
    strings = re.search(regex, row).groups()
    (row_id, col_id, real, image) = [t(s) for t, s in zip((int, int, float, float), strings)]
    lind_cpp[row_id, col_id] = real + 1j * image
lind_diff = la.norm(lind_cpp - lind_dense)
lind_diff_conj = la.norm(lind_cpp - np.conjugate(lind_dense))
print(f'lind_diff: {lind_diff}')
print(f'lind_diff_conj: {lind_diff_conj}')

rho_it = nk.exact.steady_state(lind, method="iterative", sparse=True, tol=1e-10)

with open(f"{cpp_path}/rho_mtx_ns({N})_seed({cpp_seed})_diss({dt}_0.0000_{gamma:0.4f})_prm({W:0.4f}_{U:0.4f}_{J:0.4f}).txt") as f:
    content = f.read().splitlines()
rho_cpp = np.zeros(hi.n_states * hi.n_states, dtype='complex64')
for row_id, row in enumerate(content):
    regex = '^\(([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?),([+-]?\d+(?:\.\d*(?:[eE][+-]?\d+)?)?)\)$'
    strings = re.search(regex, row).groups()
    (real, image) = [t(s) for t, s in zip((float, float), strings)]
    rho_cpp[row_id] = real + 1j * image
rho_cpp = rho_cpp.reshape((hi.n_states, hi.n_states))
rho_diff = la.norm(rho_cpp - rho_it)
rho_diff_conj = la.norm(rho_cpp - np.conjugate(rho_it))
print(f'rho_diff: {rho_diff}')
print(f'rho_diff_conj: {rho_diff_conj}')

#rho_ed = nk.exact.steady_state(lind)

#ololo = 1

