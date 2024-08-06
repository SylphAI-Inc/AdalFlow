.. _parameter:

Parameter
====================


There are two types of parameters:

* once-off, such as loss, y_pred, intermedia response where each run it will create a new one, and they are temporary and will be released after the run.
* persistent, the parameters that we are optimizing, such as those with an actual type assigned `param_type` in the `Parameter` class.


TODO: a DAG to show this.


All our targing parameter to train will end up being a leaf node in the auto-diff DAG.

In each run, the persistent parameters can be used by multiple successors if batch_size > 1. This results it to accumulate all traces.

auto-diff
-----------
To support auto-diff, we added `predecessors` , and adalflow created `peers` concept to ensure training parameters are aware of each other to not conflicting while optimizing.

For instance, if no peers context, the system instruction can generate examples or enforce output format while users want to train each of them separately.

These characteristics are important. this means for passing the score
