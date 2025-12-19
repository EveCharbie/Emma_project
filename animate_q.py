import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import math

import matplotlib.pyplot as plt
from pyorerun import BiorbdModel, PhaseRerun

import os
from pathlib import Path
import pickle

athlete_num = 1
mode = "base"  # "retroversion"  # "anteversion"  # "base"

CURRENT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = f"{CURRENT_DIR}/ocp/results/athlete_{athlete_num:03d}"
modelname = f"{CURRENT_DIR}/models/biomod_models/athlete_{athlete_num:03d}_deleva.bioMod"
model = BiorbdModel(modelname)

file_pkl = f"{RESULTS_DIR}/athlete{athlete_num:03d}_{mode}_CVG.pkl"

if not os.path.exists(file_pkl) or os.path.getsize(file_pkl) == 0:
    raise FileNotFoundError(f"Fichier absent ou vide: {file_pkl}")

with open(file_pkl, "rb") as f:
    data = pickle.load(f)

qs    = data["q_all"]
qdots = data["qdot_all"]
taus  = data["tau_all"]
time  = data["time_all"]
n_shooting = data["n_shooting"]


def plot_all_dofs(modelname: str, time: np.ndarray, data: np.ndarray,
                  type: str = 'states',
                  title: str = "Generalized coordinates (q)"):
    """
    time: shape (T,)       — temps
    q:    shape (n_q, T)   — états q
    biomod: chemin vers le .bioMod (pour lire les noms de DoF)
    """
    model = BiorbdModel(modelname)
    names = model.dof_names
    n_q = data.shape[0]

    # grille carrée ~√n

    ncols = math.ceil(math.sqrt(n_q))
    nrows = math.ceil(n_q / ncols)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(3.8*ncols, 2.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_q):
        ax = axes[i]
        if type is 'states':
            ax.plot(time, data[i, :])
        elif type is 'controls':
            ax.step(time[0:-6:6], data[i, :])

        ax.set_title(names[i], fontsize=9)
        ax.grid(True, alpha=0.3)

    # masquer les axes vides si la grille n’est pas pleine
    for j in range(n_q, nrows*ncols):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    axes[max(0, n_q-1)].set_xlabel("Time [s]")
    plt.show()

plot_all_dofs(modelname, time,qs, type='states', title="q")
plot_all_dofs(modelname, time,qdots,type='states',title="qdot")
plot_all_dofs(modelname, time, taus, type='controls',title="tau")


viz = PhaseRerun(time)
viz.add_animated_model(model, qs)
viz.rerun("swing")


