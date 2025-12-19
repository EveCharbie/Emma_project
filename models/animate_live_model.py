"""
This files allows to see the model used in the simulation.
"""
from pathlib import Path
from pyorerun import LiveModelAnimation


athlete_number = 1

current_dir = Path(__file__).parent
model_path = f"{current_dir}/biomod_models/athlete_{athlete_number:03d}_deleva.bioMod"

animation = LiveModelAnimation(model_path, with_q_charts=True)
animation.rerun()
