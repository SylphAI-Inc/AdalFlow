Parser
=============

In this note, we will explain LightRAG parser and output parsers.

Context
----------------

LLMs output text in string format.
Parsing is the process of `extracting` and `converting` the string to desired data structure per the use case.

This desired data structure can be:

- simple data types like string, int, float, boolean, etc.
- complex data types like list, dict, or data class instance.
- Code like Python, SQL, html, etc.

It honestly can be converted to any kind of formats that are required by the use case.
It is an important step for the LLM applications to interact with the external world.
Parsing is like the `interpreter` of the LLM output.

Scope and Design
------------------

Right now, we aim to cover the simple and complext data types but the code.


Converting string to structured data is similar to the step of deserialization in serialization-deserialization process.
We already have powerful ``DataClass`` to handle the serialization-deserialization for data class instance.

Parser in Action
------------------

Parser builts on top of that


Output Parsers in Action
--------------------------

Evaluate Format following
--------------------------
