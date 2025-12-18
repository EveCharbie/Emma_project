# Emma_project

## Dependencies
```bash
conda create --name emma_project python=3.13
conda install -c conda-forge pip pyqt pyqtgraph matplotlib numpy pytest black ezc3d rerun-sdk=0.21.0 trimesh tk imageio imageio-ffmpeg biorbd lxml plotly
pip install gitpython
```

### Modeling
- [create_a_population_of_models.py](models/create_a_population_of_models.py) create models with different anthropometries.
- [aminate_live_model.py](aminate_live_model.py) display a live animation of a model using pyrerun.

### Optimal Control
- [animate_q.py](animate_q.py) display a pyrerun animation of the optimal solutions.
- 
