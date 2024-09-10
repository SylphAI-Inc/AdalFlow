.. _apis-optim:

Optimization
==============



Base Classes and Data Structures
----------------------------------
The ``GradComponent`` and ``LossComponent`` are a subclass from ``Component`` to serve the purpose to differentiate the gradient and loss components in the optimization process.
And it will be used if users want to implement their own with more customization.

.. autosummary::


   optim.parameter
   optim.optimizer
   optim.grad_component
   optim.loss_component
   optim.types


   .. optim.sampler
   .. optim.llm_optimizer



.. toctree::
   :maxdepth: 1
   :hidden:

   optim.parameter
   optim.optimizer
   optim.grad_component
   optim.loss_component
   optim.types

   .. optim.sampler

Few Shot Optimizer
----------------------------

.. autosummary::

   optim.few_shot.bootstrap_optimizer

.. toctree::
   :maxdepth: 1
   :hidden:

   optim.few_shot

Textual Gradient
----------------------------
.. autosummary::

   optim.text_grad.llm_text_loss
   optim.text_grad.text_loss_with_eval_fn
   optim.text_grad.ops
   optim.text_grad.tgd_optimizer

.. toctree::
   :maxdepth: 1
   :hidden:

   optim.text_grad

Trainer and AdalComponent
----------------------------
.. autosummary::


   optim.trainer.adal
   optim.trainer.trainer

.. toctree::
   :maxdepth: 1
   :hidden:


   optim.adal
   optim.trainer


Overview
---------------

.. automodule:: optim
   :members:
   :undoc-members:
   :show-inheritance:

.. make sure it follows the same order as optim.rst
