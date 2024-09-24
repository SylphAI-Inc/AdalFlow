Multi-hop RAG Optimization
============================


question: How many storeys are in the castle that David Gregory inherited?

query 0: Number of storeys in the castle inherited by David Gregory

Add context from retriever -> query generator


query 1: Kinnairdy Castle storeys OR floors OR levels

So eventually the multi-hop RAG with the multi-hop retriever that combines the advanced Generator to transform the query into multiple querires (similar to REACT agent design)
. By knowing the castle name from the first query and retrieval (not seen from the question itself), the second time it will be able to retrieve the right context the second time.
Of course, we can even let the LLM workflow decide to stop the retrieval once it has obtained enough information.


When multi-hop is not enabled, the vanilla rag failed to give the answer.
When it is enabled, the answer is correct.

resoning:

David Gregory inherited Kinnairdy Castle, which is a tower house having five storeys and a garret, located two miles south of Aberchirder, Aberdeenshire, Scotland.

answr: Kinnairdy Castle has five storeys."
