Agent
====================

.. epigraph::

    “An autonomous agent is a system situated within and a part of an environment that senses that environment and acts on it, over time, in pursuit of its own agenda and so as to effect what it senses in the future.”

    -- Franklin and Graesser (1997)

Agents are LLM-based and themselves belong to another popular family of LLM applications besides of the well-known RAGs.
The key on Agents are their ability to reasoning, plannning, and acting via accessible tools.
In LightRAG, agents are simply a generator which can use tools, take multiple steps(sequential or parallel ) to complete a user query.



.. admonition:: References
   :class: highlight

   1. A survey on large language model based autonomous agents: https://github.com/Paitesanshi/LLM-Agent-Survey
   2. ReAct: https://arxiv.org/abs/2210.03629


.. admonition:: API References
   :class: highlight

   - :class:`components.agent.react.ReactAgent`
