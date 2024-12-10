.. _trainer:

Trainer
================
Coming soon!

Diagnose mode

A pipeline can consist of multiple generators or retrievers. Each


Computation graph
-------------------
We design two types of graphs:

1. with a simple node-graph with consistent naming of each generator(component_name or automated name by the recursive tracing (need to be consistent eventually)) [Call it thumbnail] or a better name.
2. with details for debugging and building the pipeline.

EvalFunction + Score(s)
------------------------
Currently we can assume we only support one eval_score, but eventually we need to suppport two scores, such as in the case of the multi-hop RAG.
The last llm call will have one score, and the previous two generators can potentially have two scores. One is from the last score, and the second will be from the output of the multi-hop retriever.

So, we need to assign a unique and global component id/name. [Score, component_id, component_name]

Observability
------------------------
Building blocks include: `GeneratorCallLogger`, `RetrieverCallLogger`, `LossCallLogger` where each only traces a single component.

In `AdalComponnet`, `configure_callbacks` we need both `_auto_generator_callbacks` and `_auto_retriever_callbacks` to be able to trace the call of each component.

..code-block:: python

    for name, generator in all_generators:
        call_logger = GeneratorCallLogger(save_dir=save_dir)
        call_logger.reset()
        call_logger.register_generator(name)
        logger_call = partial(call_logger.log_call, name)
        generator.register_callback(
            "on_complete", partial(_on_completion_callback, logger_call=logger_call)
        )
        file_path = call_logger.get_log_location(name)
        file_paths.append(file_path)
        log.debug(f"Registered callback for {name}, file path: {file_path}")


so when tracing, the `logger_metadata.json` will look like this:

.. code-block:: json

    {
    "retriever.query_generators.0": "/Users/liyin/.adalflow/ckpt/MultiHopRAGAdal/diagnose_train/retriever.query_generators.0_call.jsonl",
    "retriever.query_generators.1": "/Users/liyin/.adalflow/ckpt/MultiHopRAGAdal/diagnose_train/retriever.query_generators.1_call.jsonl",
    "llm": "/Users/liyin/.adalflow/ckpt/MultiHopRAGAdal/diagnose_train/llm_call.jsonl"
    }

TODO:
- [ ] support multiple eval scores.
- [ ] logger meta data

  {
    "llm": "/Users/liyin/.adalflow/ckpt/MultiHopRAGAdal/diagnose_train/llm_call.jsonl"
}
- [ ] retriever log: call_logger = GeneratorCallLogger(save_dir=save_dir)
