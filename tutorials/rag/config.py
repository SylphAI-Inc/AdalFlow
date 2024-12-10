configs = {
    "embedder": {
        "batch_size": 100,
        "model_kwargs": {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        },
    },
    "retriever": {
        "top_k": 2,
    },
    "generator": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.3,
        "stream": False,
    },
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 400,
        "chunk_overlap": 200,
    },
}
