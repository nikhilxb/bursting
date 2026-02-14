# Usage

# Single-unit experiments

The `analyze.py` script run the analysis and plotting for the biophysical and simplified models. You can specify the configuration file for each model using the `--config-name` argument and the path to a config from `configs`. For example:

```bash
python analyze.py --config-name biophysical/constant.yaml
python analyze.py --config-name simplified/constant.yaml
```

The outputs will be logged to the `$OUTPUT_ROOT` directory.

# Circuit experiments
