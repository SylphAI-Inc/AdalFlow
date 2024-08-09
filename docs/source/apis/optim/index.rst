.. _apis-optim:

Optimizing
==============



Base Classes
----------------------------
.. autosummary::


   optim.parameter
   optim.optimizer


   .. optim.sampler
   .. optim.llm_optimizer



.. toctree::
   :maxdepth: 1
   :hidden:

   optim.parameter
   optim.optimizer

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

   optim.text_grad.function
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
