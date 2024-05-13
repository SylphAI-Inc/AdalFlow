# Step 1: read processed data
# (optional) step 2: load json data for current firm (to merge with)
# Step 2: load all documents to extract information from for the target url and firm

# LocalDocumentDB: name = "url"
# Document: text -> {tag_to_contet: {tag: content}} (content can be any type)

# Components
# 1. Generator for url filter. Input: all documents meta_data["url"], Output: List[str]
# All fields to be extracted: as inputs, multiple pipelines
# initiate the pipeline for each field
# Step1: filter out uncessary documents by urls. Component URLFilter(go through all documents, meta_data["url"] == url),
#      Callable to extract urls into a string as query_str using the meta_data["url"] (similar to dataset from pytorch, need batch if the data is too big)
#      The usage of Pytorch module's apply

# Step2: filter out Document from the local db by meta_data

# Step3: split data, embed, and save to another local db, and attach FAISSRetriever to the local db and generate [context_str]

# So local database needs to apply filter

# Step4: initiate extraction task for each field
# Three types of retriever for the generators
# One: document retriever and generator with combine.
# 1. documents_retriever input documents are just all urls. LLMs as Retriever````
# 2. convert all retrieved documents to context_str -> Generator1, GenetatorContinue (loop) -> multiple responses and then combine

# Two: dense retriever + query expansion + generator
# typical FAISSRetriever, Ensure if the query fails due to token limits, we have auto retry with simple truncation (error and robustness handling)
