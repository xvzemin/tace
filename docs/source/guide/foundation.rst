Foundational Model
==================

This tutorial demonstrates how to **load a pretrained foundational model** 
and attach it as an ASE calculator.

For more advanced topics—such as fine-tuning, or alternative interfaces, 
please refer to the corresponding tutorials.

Model Selection
---------------

For most simulations, start with ``TACE-OMat24-7M``. Choose a TECE RRA model
when accuracy takes priority over speed and memory use.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Priority
     - Recommended models
     - Indicative system size
   * - Accuracy
     - ``TECE-OMat24-RRA-1.0``, ``TECE-OAM-RRA-1.0``
     - About 1,000 atoms in one 80GB GPU
   * - Accuracy, speed, and memory balance
     - ``TACE-OMat24-7M``, ``TACE-OAM-7M``
     - About 8,000 atoms in one 80GB GPU
   * - General-purpose materials simulations
     - ``TACE-OAM-L``
     - About 3,000 atoms

Enable a supported backend as described in :ref:`acceleration-tutorial` 
when comparing performance.

Model Overview
--------------

TACE uses atomic cluster expansion; TECE additionally uses edge cluster
expansion. ``RRA`` denotes radial rotary attention. ``OMat24`` models are
trained on OMat24, whereas ``OAM`` models are subsequently trained on sAlex
and MPtrj. The arrow below indicates this training sequence.

The materials models cover 89 elements at the PBE+U level of theory.
``TACE-v1-LES-REICO-5-PdAgCHO`` targets heterogeneous catalysis with
Pd, Ag, C, H, and O at the PBE level of theory.

.. list-table::
   :header-rows: 1
   :widths: 40 10 30 20

   * - Model
     - Size
     - Training data
     - Required TACE version
   * - ``TACE-OMat24-7M``
     - M
     - OMat24
     - ``>=0.2.0``
   * - ``TACE-OAM-7M``
     - M
     - OMat24 → sAlex + MPtrj
     - ``>=0.2.0``
   * - ``TECE-OMat24-RRA-1.0``
     - XL
     - OMat24
     - ``>=0.2.0``
   * - ``TECE-OAM-RRA-1.0``
     - XL
     - OMat24 → sAlex + MPtrj
     - ``>=0.2.0``
   * - ``TACE-OMat24-RRA-1.0``
     - XL
     - OMat24
     - ``>=0.2.0``
   * - ``TACE-OAM-RRA-Preview``
     - XL
     - OMat24 → sAlex + MPtrj
     - ``>=0.2.0``
   * - ``TACE-OMat24-L``
     - L
     - OMat24
     - ``>=0.2.0``
   * - ``TACE-OAM-L``
     - L
     - OMat24 → sAlex + MPtrj
     - ``>=0.2.0``
   * - ``TACE-v1-OMat24-M``
     - M
     - OMat24
     - ``==0.1.0``
   * - ``TACE-v1-OAM-M``
     - M
     - OMat24 → sAlex + MPtrj
     - ``==0.1.0``
   * - ``TACE-v1-LES-REICO-5-PdAgCHO``
     - M
     - REICO-5-PdAgCHO
     - ``==0.1.0``

Model Download and Cache
------------------------

When loading a model through ``from tace.foundations import tace_foundations``, 
the pretrained weights will be **downloaded automatically** and cached locally.

By default, all models are stored under::

    ~/.cache/tace/

If your network connection is unstable or restricted, you may manually
download the pretrained models from the
`TACE model collection on Hugging Face <https://huggingface.co/xvzemin/tace-foundations/tree/main>`_.
The model weights are distributed under CC BY 4.0.

For automatic loading, place the checkpoint directly under ``~/.cache/tace/``
using the filename expected by the registry. For example,
``TACE-OAM-7M`` uses ``~/.cache/tace/TACE-OAM-7M.pt``.
To list the registry keys supported by your installation:

.. code-block:: python

    from tace.foundations import tace_foundations

    print(tace_foundations.list_models())

For a release not listed in the registry, download its checkpoint from
Hugging Face and pass its local path as ``model`` to ``TACEAseCalc``.

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

Honors and Milestones
---------------------

* **2026-07-08 — Matbench Discovery SOTA.** ``TECE-OAM-RRA-1.0`` ranked
  first on the default `Matbench Discovery leaderboard
  <https://matbench-discovery.materialsproject.org/>`_ by Combined Performance
  Score (CPS), with a score of 0.908.

.. figure:: ../../../fig/matbench_tece_rra.png
   :alt: Matbench Discovery ranking on July 8, 2026, led by TECE-OAM-RRA-1.0 with a CPS of 0.908.
   :width: 100%

   Matbench Discovery default ranking as of July 8, 2026.
