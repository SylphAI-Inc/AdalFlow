A Guideline on LLM Evaluation
============

Evaluating LLMs and their applications is crucial for understanding their capabilities and limitations. Overall, this evaluation is a complex and multifaceted process. Below, we provide a guideline for evaluating LLMs and their applications, incorporating aspects outlined by *Chang et al.* [1]_:

* **What to evaluate**: the tasks and capabilities that LLMs are evaluated on.
* **Where to evaluate**: the datasets and benchmarks that are used for evaluation.
* **How to evaluate**: the protocols and metrics that are used for evaluation.


What to evaluate?
------------------------------------------
When we are considering the LLM evaluation, the first question that arises is what to evaluate. Deciding what tasks to evaluate or which capabilities to assess is crucial, as it influences both the selection of appropriate benchmarks (where to evaluate) and the choice of evaluation methods (how to evaluate). Below are some commonly evaluated tasks and capabilities of LLMs:

* *Natural language understanding* (NLU) tasks, such as text classification and sentiment analysis, which evaluate the LLM's ability to understand natural language.
* *Natural language generation* (NLG) tasks, such as text summarization, translation, and question answering, which evaluate the LLM's ability to generate natural language.
* *Reasoning* tasks, such as mathematical, logic, and common-sense reasoning, which evaluate the LLM's ability to perform reasoning and inference to obtain the correct answer.
* *Robustness*, which evaluate the LLM's ability to generalize to unexpected inputs.
* *Fairness*, which evaluate the LLM's ability to make unbiased decisions.
* *Domain adaptation*, which evaluate the LLM's ability to adapt from general language to specific new domains, such as medical or legal texts, coding, etc.
* *Agent applications*, which evaluate the LLM's ability to use external tools and APIs to perform tasks, such as web search.

For a more detailed and comprehensive description of the tasks and capabilities that LLMs are evaluated on, please refer to the review papers by *Chang et al.* [1]_ and *Guo et al.* [2]_.

Where to evaluate?
------------------------------------------


How to evaluate?
------------------------------------------

.. [1] Chang, Yupeng, et al. "A survey on evaluation of large language models." ACM Transactions on Intelligent Systems and Technology 15.3 (2024): 1-45.
.. [2] Guo, Zishan, et al. "Evaluating large language models: A comprehensive survey." arXiv preprint arXiv:2310.19736 (2023).
