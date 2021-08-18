import jax
import numpy as np
from netket.hilbert import Fock


hi = Fock(n_max=1, n_particles=4, N=8)
# Hilbert space info
all_states = hi.all_states()
numbers_to_states = hi.numbers_to_states(np.array([0, 1, 3, 68, 69]))
rng_k1, rng_k2 = jax.random.split(jax.random.PRNGKey(1))
random_states = hi.random_state(key=rng_k1, size=10)
states_to_numbers = hi.states_to_numbers(np.asarray([[1, 1, 1, 1, 0, 0, 0, 0]]))
