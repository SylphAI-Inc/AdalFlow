Parser
=============
In this note, we will explain LightRAG parser and output parsers.

Context
----------------

Parser
----------------
LLMs output text in string format.
Parser is a component used to parse that string into desired data structure per the use case.

Converting string to structured data is similar to the step of deserialization in serialization-deserialization process.
We already have powerful ``DataClass`` to handle the serialization-deserialization for data class instance.
Parser builts on top of that
