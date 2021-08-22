import pandas as pd
import plotly.graph_objects as go
from ndm.plot.layout import add_layout
from ndm.plot.save import save_figure


# Model params
model = 'ising1d'
L = 6
gp = 0.3
Vp = 2.0

path = f"/media/sf_Work/dl/netket/{model}/L({L})_V({Vp})_g({gp})"

# Ansatz params

params = [
    {"alpha": 2, "beta": 2, 'n_samples': 1000, 'n_samples_diag': 1000},
    #{"alpha": 2, "beta": 2, 'n_samples': 2000, 'n_samples_diag': 2000},
    #{"alpha": 2, "beta": 2, 'n_samples': 1000, 'n_samples_diag': 4000},
    {"alpha": 2, "beta": 2, 'n_samples': 4000, 'n_samples_diag': 1000},
    #{"alpha": 4, "beta": 4, 'n_samples': 4000, 'n_samples_diag': 1000},
]

fig_ldagl = go.Figure()
fig_diff_norm = go.Figure()
for p in params:
    df = pd.read_excel(f"{path}/metrics_size({p['alpha']}_{p['beta']})_samples({p['n_samples']}_{p['n_samples_diag']}).xlsx")

    fig_ldagl.add_trace(
        go.Scatter(
            x=df['iteration'].to_list(),
            y=df['ldagl_mean'].to_list(),
            showlegend=True,
            name=f"alpha={p['alpha']} beta={p['beta']} nsamp={p['n_samples']} nsampdiag={p['n_samples_diag']}",
            mode="lines",
        )
    )

    fig_diff_norm.add_trace(
        go.Scatter(
            x=df['iteration'].to_list(),
            y=df['norm_rho_diff_1'].to_list(),
            showlegend=True,
            name=f"alpha={p['alpha']} beta={p['beta']} nsamp={p['n_samples']} nsampdiag={p['n_samples_diag']}",
            mode="lines",
        )
    )

add_layout(fig_ldagl, "Iterations", r"$L^\dagger L$", f"Ising 1D  (L={L}, V={Vp}, g={gp})")
fig_ldagl.update_layout({'colorway': ['red', 'blue', 'green', 'orange', 'brown']})
fig_ldagl.update_yaxes(type="log")
save_figure(fig_ldagl, f"{path}/ldagl")


add_layout(fig_diff_norm, "Iterations", r"$\left\Vert \rho^{\text{exact}} - \rho^{\text{neural}} \right\Vert$", f"Ising 1D (L={L}, V={Vp}, g={gp})")
fig_diff_norm.update_layout({'colorway': ['red', 'blue', 'green', 'orange', 'brown']})
fig_diff_norm.update_yaxes(type="log")
save_figure(fig_diff_norm, f"{path}/diff_norm")
