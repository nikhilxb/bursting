# Installation

## 1. Conda virtual environment

Setup the `conda` virtual environment to manage depencencies.


Install project dependencies from the configuration YAML file:

```bash
conda env create --file .env/conda.yaml --solver libmamba

# Alternative: If already created environment and want to update with new YAML file.
conda env update --file .env/conda.yaml --solver libmamba
```

Set up the project environment variables:

```bash
# 1. Manually edit "env/conda-vars.sh" with the correct paths on your machine.
vim .env/conda-vars.sh

# 2. Initialize the variables.
source .env/conda-setup.sh

# 3. Reactivate environment and check variables.
conda activate <$CONDA>
conda env config vars list
```

## 2. Pip libraries

Install additional Python libraries via `pip`:

```bash
pip install -r .env/requirements.txt
```
