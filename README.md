![tests](https://github.com/Raylend/agnprocesses/actions/workflows/pr.yml/badge.svg)

# `agnprocesses` — AGN processes modelling toolkit

## Installation (Ubuntu based distribution)

1. Clone this repo

```bash
git clone https://github.com/Raylend/agnprocesses.git
cd agnprocesses
```

2. (Recommended) Create virtual environment for the library

```bash
python3 -m venv agnenv
source agnenv/bin/activate
```

3. Install dependencies (**TBD: list dependencies in setup script**)

```bash
pip install -r requirements.dev.txt
```

4. Compile shared libraries and install `agnprocesses` python package

```bash
make install
```

5. Compiled libraries are currently placed at `./bin/shared` and you must define LD_LIBRARY_PATH environment variable to use them. This may be done in `~/.bashrc` or in any other startup script (e.g. in `agnenv/bin/activate`). (**TBD: install libs in /usr/local/lib or in user-specified location**)

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

See `examples` for notebooks with example usage.

