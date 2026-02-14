# Infra

This library provides infrastructure for experiment management.


### `launcher`: Hydra configuration and data organization
The launcher is built atop Hydra, a framework for modularly composing nested configurations and sweeping over configuration parameters (https://hydra.cc/). As such, Hydra is a powerful tool for managing the complexity of highly nested/configurable systems and rapidly running experiments on different variants.

However, Hydra imposes minimal constraints on how configs and runtime outputs are organized, which enables inconsistent and suboptimal practices between experiments and projects. Hydra also does not include out-of-the-box support for launching jobs in a computing cluster and/or containerized environment, which is common practice for large-scale experiments.

The launcher provides several features:

1. **Defines a base Hydra configuration** that sets up cluster/container support and a consistent organization of output directories.
2. **Registers some custom OmegaConf resolvers** for more powerful interpolations.

The launcher can be easily integrated through 2 steps:

1. In a Hydra main script (e.g. `train.py`), import the module to register various plugins/callbacks that extend Hydra's functionality:
    ```python
    import infra.launcher  # Import this.
    ```

2. In a Hydra experiment config (e.g. `train.yaml` that [configures an experiment](https://hydra.cc/docs/1.3/patterns/configuring_experiments/) in `@package __global__`), extend the base infrastructure config:
    ```yaml
    # @package __global__

    defaults:
      - /infra/base  # Extend this.
      - /foo/bar@baz: option1
      # ...
      - _self_
    ```
