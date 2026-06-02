# GlobalPSRO

This is the code for **Global Policy-Space Response Oracles for Two-Player Zero-Sum Games**.

## Included Algorithms

- Standard PSRO
- PSD-PSRO
- NeuPL
- NaiveGlobalPSRO, a naive implementation of GlobalPSRO where each candidate is trained and evaluated independently
- SimplexGlobalPSRO, the GlobalPSRO implementation reported in the paper

## Install

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

PyTorch is installed separately so that the CUDA build can match the machine. Install the appropriate `torch==2.0.0` build inside the activated environment. Example CUDA 11.7 command:

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
```

Then install this package and its runtime dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

## Run Scripts

Shell entrypoints are under:

```bash
onpolicy/scripts/train_poker_scripts/
onpolicy/scripts/train_goofspiel_scripts/
```

Example:

```bash
cd onpolicy/scripts/train_poker_scripts
./train_poker_SimplexGlobalPSRO.sh
```

