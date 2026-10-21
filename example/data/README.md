BaTiO3.xyz, from https://github.com/mir-group/allegro-pol
TS2-AA-on-phase-O5.vasp, A transition state
toluene from https://github.com/openmm/openmm-ml/tree/main/test/data

HORM full Hessians: install `kagglehub>=1.0.2`, then run `python download_HORM.py`
without arguments. It downloads the author's latest Kaggle release to
`~/dataset/HORM/raw/v<version>` and converts the official splits to
`~/dataset/HORM/{train,valid,test}.db`. Original units are preserved.
Source: https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm
