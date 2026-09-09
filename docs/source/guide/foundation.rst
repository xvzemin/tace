Foundational Model
==================

This tutorial demonstrates how to **load a pretrained foundational model** and attach it as an ASE calculator.

For more advanced topics—such as fine-tuning, or alternative interfaces, please refer to the corresponding dedicated tutorials.

Model Download and Cache
------------------------

When loading a model through ``from tace.foundations import tace_foundations``, the pretrained weights will be
**downloaded automatically** and cached locally.

By default, all models are stored under::

    ~/.cache/tace/

If your network connection is unstable or restricted, you may manually
download the pretrained models from the
`TACE Foundation Models repository <https://github.com/xvzemin/tace-foundations>`_.

After downloading, please **keep the original directory and file names unchanged**
and place them directly under ``~/.cache/tace/`` so that TACE can locate them correctly.

Minimal ASE Example
-------------------

Below is a minimal working example showing how to use a TACE Foundational Model
as an ASE calculator:

.. code-block:: python

    import torch
    from ase.io import read
    from tace.foundations import tace_foundations
    from tace.interface.ase import TACEAseCalc, add_dispersion

    # Load a pretrained foundational model
    # The model will be auto-downloaded to ~/.cache/tace if not present
    model = tace_foundations["TACE-OAM-7M"]

    dtype = "float32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fidelity fidelity_idx (0 corresponds to the first fidelity)
    fidelity_idx = 0

    atoms = read("../unrelaxed.xyz", index=0)
    calc = TACEAseCalc(
        model=model,
        dtype=dtype,
        device=device,
        fidelity_idx=fidelity_idx,
    )
    atoms.calc = calc

Dispersion Correction (Optional)
--------------------------------

Dispersion interactions can also be supported by calling third-party libraries.
For detailed instructions, see ase guide.

