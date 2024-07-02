default_config = {
    "document_splitter": {
        "component_name": "DocumentSplitter",
        "component_config": {
            "split_by": "word",
            "split_length": 400,
            "split_overlap": 200,
        },
    },
    "to_embeddings": {
        "component_name": "ToEmbeddings",
        "component_config": {
            "embedder": {
                "component_name": "Embedder",
                "component_config": {
                    "model_client": {
                        "component_name": "OpenAIClient",
                        "component_config": {},
                    },
                    "model_kwargs": {
                        "model": "text-embedding-3-small",
                        "dimensions": 256,
                        "encoding_format": "float",
                    },
                },
            },
            "batch_size": 100,
        },
    },
}
