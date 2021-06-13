# `agnprocesses` — AGN processes modelling toolkit

## Installation (Ubuntu based distribution)

1. Clone this repo

```bash
git clone https://github.com/Raylend/agnprocesses.git
```

2. Install dependencies (**TBD: list dependencies in setup script**)

```bash
pip install -r requirements.txt
```

3. Install libraries and python package

```bash
make install
```

4. Compiled libraries are currently placed at `./bin/shared`. You must define LD_LIBRARY_PATH environment variable to use them. This may be done in `~/.bashrc` or in other kind of startup script (e.g. in venv activation script). (**TBD: install libs in /usr/local/lib or in user-specified location**)

```bash
echo "
export LD_LIBRARY_PATH=$(pwd)/bin/shared:\$LD_LIBRARY_PATH" >> your-activation-script
```

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

See `examples/main.py` as an example of the SSC (synchrotron self Compton) model (**other examples TBD**).

