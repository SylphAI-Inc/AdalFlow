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

Other logs:

-----------------

1. fix the dspy code  at `.venv/lib/python3.12/site-packages/dsp/modules/colbertv2.py`

.. code-block::python

    from tenacity import retry, stop_after_attempt, wait_exponential


    @CacheMemory.cache
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def colbertv2_get_request_v2(url: str, query: str, k: int):
        assert k <= 100, "Only k <= 100 is supported for the hosted ColBERTv2 server."

        payload = {"query": query, "k": k}

        try:
            res = requests.get(url, params=payload, timeout=10)
            res.raise_for_status()
            response_json = res.json()

            # Check for an error in the response.
            if response_json.get("error"):
                raise ConnectionError(f"Error from server: {response_json['message']}")

            # If we get a valid 'topk' response, return immediately.
            if "topk" in response_json:
                topk = response_json["topk"][:k]
                return [{**d, "long_text": d["text"]} for d in topk]

        except requests.exceptions.Timeout:
            raise TimeoutError("The request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Request failed: {e}")

        raise KeyError("'topk' key not found in the response.")

2. If error in diagnose similar to

   ..code-block:: python

        Error loading jsonl file /Users/liyin/.adalflow/ckpt/MultiHopRAGAdal/diagnose_train/llm_call.jsonl: line contains invalid json: unexpected content after document: line 1 column 8568 (char 8567) (line 62)
    Traceback (most recent call last):
    File "/Users/liyin/Documents/test/LightRAG/benchmarks/hotpot_qa/adal_exp/train_multi_hop_rag.py", line 153, in <module>
        train_diagnose(**gpt_3_model)
    File "/Users/liyin/Documents/test/LightRAG/benchmarks/hotpot_qa/adal_exp/train_multi_hop_rag.py", line 97, in train_diagnose
        trainer.diagnose(dataset=trainset, split="train")
    File "/Users/liyin/Documents/test/LightRAG/adalflow/adalflow/optim/trainer/trainer.py", line 228, in diagnose
        sorted_logs = [logs_dict[id] for id in sorted_ids]
                    ~~~~~~~~~^^^^
    KeyError: '5a8b57f25542995d1e6f1371'

You can go to the `llm_call.jsonl` file and clean all content of the file. Then rerun the training script.
