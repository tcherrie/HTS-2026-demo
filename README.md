# HTS-2026

[![GitHub license](https://img.shields.io/github/license/houtazak/HTS-2026)](https://github.com/houtazak/HTS-2026) [![GitHub release](https://img.shields.io/github/release/houtazak/HTS-2026.svg)](https://github.com/houtazak/HTS-2026/releases/) [![GitHub stars](https://img.shields.io/github/stars/houtazak/HTS-2026)](https://github.com/houtazak/HTS-2026/stargazers)
[![DOI](https://zenodo.org/badge/1157299604.svg)](https://zenodo.org/doi/10.5281/zenodo.18634454)

This repository contains FEM implementations of H-formulation for 2D and 3D High Temperature Superconductors (HTS) simulations using NGSolve, presented at HTS workshop 2026.

## 1) Quickstart

Click here to open the JupyterLite environment :

[![lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://tcherrie.github.io/HTS-2026/lab/?path=index.ipynb)

## 2) Installation
Install the required packages in a new Python environment (version 3.13 or later, see `requirements.txt`). 

For doing so, open a terminal, create and activate a dedicated Python environment, using for instance `conda` (requires the installation of [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) ): 

`conda create -n myenv python=3.13`

with `myenv` being replaced by your environment name, and activate it:

`conda activate myenv`

Then, go to the folder where the code is located:

`cd C:\path\to\the\folder`

replacing `C:\path\to\the\folder` by the path to the local folder containing this code. The, install the required packages using 

`pip install -r requirements.txt`

After that, you should be able to run the scripts on your computer.

### Run the scripts
Execute one of the scripts in your favorite IDE within your newly created `myenv` environment:
- `2D_tape_external_field.py`
- `2D_tape_transport_current.py`
- `3D_bulk_external_field.py`

## 2) Contents of the repository

```
.
├── results_COMSOL/  # Reference results
│   ├── 2D/
│   │   ├── external_field/
│   │   │   ├── AC_Losses_1mT.txt
│   │   │   ├── ...
│   │   └── transport_current/
│   │       ├── AC_Losses_22.4A.txt
│   │       ├── ...
│   └── 3D/
│       ├── AC_Losses_5mT.txt
│       └── AC_Losses_20mT.txt
│
├── utils/  # Meshes, nonlinear solver and plot utilities
│   ├── geometry.py
│   ├── solver.py
│   ├── solver_mixed.py
|   ├── trace.py
│   ├── mesh_comsol_2D.mphtxt
│   └── mesh_comsol_3D.mphtxt
│
| # Scripts to execute
├── 2D_tape_external_field.py 
├── 2D_tape_transport_current.py
├── 3D_bulk_external_field.py
│
| # Installation and instructions
├── requirements.txt # numpy, matplotlib, ngsolve
├── README.md
|
| # Metadata
├── AUTHORS # Z. Houta, T. Cherrière, L. Quéval
├── CITATION.cff 
└── LICENSE # GNU LGPL 3.0 or any later version
```

## 3) Citation

Please use the following citation reference if you use the code:

    Z. Houta, T. Cherrière, L. Quéval. GitHub repository houtazak/HTS-2026: Initial release, v0.0. Zenodo archive: https://doi.org/10.5281/zenodo.18634454

Bibtex entry:

    @software{houtazak_2026_10718117,
    author       = {Houta, Zakaria and Cherrière, Théodore and Quéval, Loïc},
    title        = {GitHub repository houtazak/HTS-2026: Initial release},
    year         = {2026},
    publisher    = {Zenodo},
    version      = {v0.0},
    doi          = {10.5281/zenodo.18634454},
    url          = {https://doi.org/10.5281/zenodo.18634454},
    copyright    = {GNU Lesser General Public License v3.0 or any later version}
    }

## 4) License

Copyright (C) 2026 Zakaria HOUTA (zakaria.houta@centralesupelec.fr), Théodore CHERRIERE (theodore.cherriere@centralesupelec.fr), Loïc QUEVAL (loic.queval@centralesupelec.fr)


This code is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License and of the GNU General Public License along with this code. If not, see <http://www.gnu.org/licenses/>. Please read their terms carefully and use this copy of the code only if you accept them.
