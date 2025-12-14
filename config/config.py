# Embedding model for sentence transformers
# IMPORTANT: Must match the model used to create vectors in Qdrant!
# Current vectors were created with stsb-roberta-large (768 dimensions)
# If you want to use a different model, you need to recreate the vectors

# Using stsb-roberta-large to match existing vectors (768 dimensions)
EMBEDDING_MODEL = "stsb-roberta-large"  # English-only, larger model, 768 dimensions

# Alternative models (uncomment to use, but you'll need to recreate vectors):
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dimensions, multilingual
# EMBEDDING_MODEL = "intfloat/multilingual-e5-base"  # Multilingual, good for Russian, 768 dimensions

# Quantization / dtype for SentenceTransformer embeddings.
# Allowed values: "float32" (default), "float16", "float8", "int8", "int"
EMBEDDING_DTYPE = "float16"