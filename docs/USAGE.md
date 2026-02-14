# Usage

## Single-unit experiments

The `analyze.py` script runs the analysis and plotting for the single-unit experiments with the biophysical and simplified models. You can specify the experiment using the `--config-name` argument and the path to a config file from [`configs`](../configs/).

The outputs will be logged to the `$OUTPUT_ROOT` directory specificied in [`.env/conda-setup.sh`](../.env/conda-setup.sh).

For main text experiments:

```bash
python analyze.py --config-name biophysical/constant.yaml
python analyze.py --config-name simplified/constant.yaml

python analyze.py --config-name biophysical/pulse.yaml
python analyze.py --config-name simplified/pulse.yaml

python analyze.py --config-name biophysical/periodic.yaml
python analyze.py --config-name simplified/periodic.yaml
```

For supplemental experiments:

```bash
python analyze.py --config-name simplified/constant_variants_[0-5].yaml
python analyze.py --config-name simplified/pulse_variants_[0-5].yaml
python analyze.py --config-name simplified/periodic_variants_[0-5].yaml
```


## Circuit experiments

The `circuits.ipynb` notebook runs the analysis and plotting for the circuit experiments with the simplified model. The figures and data tables in the output cells replicate the results in the main text and supplements.
