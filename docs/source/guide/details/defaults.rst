defaults
========

This example shows how to configure the ``defaults`` section in ``tace.yaml`` to read multiple YAML files from different paths and merge them.

.. code-block:: yaml

    defaults:
        - _self_                        
        - config/model@model: tace      
        - config/logger@_here_: wandb

Explanation:

- ``_self_``: Start from the current YAML file itself.  
- ``config/model@model: tace``: Load model configuration from ``./config/model/tace.yaml`` and add all parameters under the ``model`` field.  
- ``config/logger@_here_: wandb``: Load logger configuration from ``./config/logger/wandb.yaml`` and merge all parameters directly into the current file.



Syntax of Hydra and OmegaConf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


For more advanced basic syntax of Hydra and OmegaConf, such as variable interpolation and simple resolvers, we will not 
cover them here. If you are interested, you can refer to the official documentation:

- Hydra: https://hydra.cc/docs/advanced/override_grammar/basic/
- OmegaConf: https://omegaconf.readthedocs.io/en/2.3_branch/usage.html
 
