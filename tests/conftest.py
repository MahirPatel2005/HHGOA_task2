import os


# Keep the test suite hermetic even when a developer's .env points at cloud services.
for _key, _value in {
    "VECTOR_BACKEND": "local",
    "EMBEDDING_BACKEND": "hash",
    "QDRANT_URL": "",
    "QDRANT_API_KEY": "",
    "GENERATION_PROVIDER": "auto",
    "OLLAMA_API_KEY": "",
    "GROQ_API_KEY": "",
    "SARVAM_API_KEY": "",
}.items():
    os.environ[_key] = _value
