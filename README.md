![tests](https://github.com/Raylend/agnprocesses/actions/workflows/pr.yml/badge.svg)

# `agnprocesses` — AGN processes modelling toolkit

## Installation (Ubuntu based distribution)

1. Clone this repo

```bash
git clone https://github.com/Raylend/agnprocesses.git
cd agnprocesses
```

2. If you have not already installed [conda](https://www.anaconda.com/products/individual) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html), please install one of them (or both). After the installation has completed, open a new terminal window in the project folder. Create an environment for the project and install dependencies via

```bash
conda env create -f requirements.dev.yml
```

or

```bash
mamba env create -f requirements.dev.yml
```

3. Activate the installed environment:

```bash
conda activate agnprocesses
```

4. Compile shared libraries and install `agnprocesses` python package

```bash
make install
```

5. Compiled libraries are currently placed at `./bin/shared` and you must define LD_LIBRARY_PATH environment variable to use them. To do so run

```bash
conda env config vars set LD_LIBRARY_PATH=$(pwd)/bin/shared
```

6. (Optional) For running electromagnetic_cascades.py module with *NVIDIA CUDA* you should additionally install pytorch following the [instructions](https://pytorch.org/get-started/locally/).

## Development

Uninstall package and clean (almoast) all generated files:

```bash
make clean
```

Clean and install in one command:

```bash
make reinstall
```

## Usage

See `examples` for notebooks with example usage.
