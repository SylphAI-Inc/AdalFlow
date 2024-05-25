This is to show how LightRAG is used to optimize a task end to end, from the using of datasets, the configuration,s the setting up of evalutor, the training pipeline on top of the `task pipeline ("Model")` itself.

Besides the auto optimzing of the task pipeline, we also show that how to use this optimized task pipeline to label more training data. And then we train a smaller classifier using embeddings + a classifier head (linear from pytorch or sklean) and train the classifier on the new labeled data.

We compare (1) classifier + llm-synthetic data, (2) classifier + ground truth data, (3) classifier + llm-synthetic data + ground truth data.

And finally you will have a classifier, cheaper and faster to run and perform the same or even better than the original llm task pipeline.
## Task pipeline(Model)
## Manual prompt engineering [ICL]
## Auto promot engineering [ICL]
## Optimizing it with model finetuning using TorchTune [Optional]
Not necessary as for a classification any LLM is an over-kill.