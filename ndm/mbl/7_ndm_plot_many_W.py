import netket as nk
import numpy as np
from numpy import linalg as la
import pandas as pd
import os
from netket.hilbert import Fock
from tqdm import tqdm
import plotly.graph_objects as go
from ndm.plot.layout import add_layout
from ndm.plot.save import save_figure

# Model params
model = 'mbl'
N = 8
seed = 1
Ws = np.linspace(0.0, 20.0, 101)
U = 1.0
J = 1.0
dt = 1
gamma = 0.1

# Ansatz params
beta = 2
alpha = 2
n_samples = 5000
n_samples_diag = 1000
n_iter = 500

np.random.seed(seed)

path = f"/media/sf_Work/dl/netket/{model}/N({N})_rnd({seed})_H(var_{U:0.4f}_{J:0.4f})_D({dt}_{gamma:0.4f})"
if not os.path.exists(f"{path}"):
    os.makedirs(f"{path}")

metrics_df = pd.read_excel(f"{path}/metrics_size({alpha}_{beta})_samples({n_samples}_{n_samples_diag}).xlsx", index_col="iteration")

columns = [f"{x}_{y:0.4f}" for x in ['norm_rho_diff'] for y in Ws]
norms = [metrics_df.loc[500, col] for col in columns]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=Ws,
        y=norms,
        showlegend=False,
        name="",
        mode="lines",
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(
                width=1
            )
        )
    )
)
add_layout(fig, r"$W$", r"$\left\Vert \rho^{\text{exact}} - \rho^{\text{neural}} \right\Vert$", r"MBL $N=8, U=1, J=1, \gamma=0.1$")
fig.update_layout({'colorway': ['red', 'blue', "red", "red"]})
save_figure(fig, f"{path}/rho_diff")