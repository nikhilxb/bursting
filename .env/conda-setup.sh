#!/bin/bash -e

# Absolute path to repo root directory (no whitespaces or trailing slash).
REPO_ROOT="$HOME/bursting"

# Absolute path to project root directory (no whitespaces or trailing slash).
# This directory will be the working directory when executing all scripts.
PROJECT_ROOT="$HOME/bursting"

# Absolute path to output root directory (no whitespaces or trailing slash).
# This directory will be where all scripts write outputs via infrastructure like Hydra.
OUTPUT_ROOT="$HOME/bursting/outputs"

# Name of the repo conda environment.
CONDA=bursting

# ==================================================================================================

# Ensure running scripts will place any packages from the libraries (custom and third-party) into
# the top-level namespace (e.g. "import mylib"). Other packages (projects and tools) will need to
# use a namespace package (e.g. "import projects.myproject").
PYTHONPATH="$PROJECT_ROOT"
PYTHONPATH="$PYTHONPATH:$REPO_ROOT/libs"
PYTHONPATH="$PYTHONPATH:$REPO_ROOT/libs3"

# Persist environment variables to conda environment config.
conda activate $CONDA
conda env config vars set REPO_ROOT=$REPO_ROOT
conda env config vars set PROJECT_ROOT=$PROJECT_ROOT
conda env config vars set OUTPUT_ROOT=$OUTPUT_ROOT
conda env config vars set CONDA=$CONDA
conda env config vars set PYTHONPATH=$PYTHONPATH
conda deactivate
