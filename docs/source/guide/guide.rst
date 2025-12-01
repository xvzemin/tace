Tutorials
=========

Before officially starting to read the following tutorial and use TACE, please ensure that the following requirements are satisfied:

- You have basis knowledge of the ``Python`` programming language.  

- You are familiar with the general workflow of ``Deep Learning`` models, such as data preprocessing and training process.  

Our model is developed based on three major Python packages: 
`PyTorch <https://pytorch.org/>`_, 
`PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_, 
and `Hydra <https://hydra.cc/>`_.

- **PyTorch** provides the fundamental components for model implementation and training, such as neural network layers, optimizers, and learning rate schedulers. 
  With its flexible API and efficient GPU support, it serves as the backbone of deep learning development.

- **PyTorch Lightning** simplifies the training process, especially for distributed and multi-GPU training. 
  Many mainstream models are now implemented with Lightning because it standardizes the training pipeline while keeping research code clean and modular.

- **Hydra** enables configuration management through ``YAML`` files. 
  It allows users to dynamically instantiate any Python class, function, or method from external packages with minimal effort. 
  This makes the entire training and evaluation pipeline highly customizable. 
  For example, users can freely use their own optimizers, learning rate schedulers, etc., without modifying the source code of the software.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   training.rst
   ase.rst
   lmp.rst
   scripts.rst