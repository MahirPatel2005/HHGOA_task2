Vaani / HH Goa Voice RAG — secure Vercel drop folder

This folder is the public static frontend only.

SAFE TO PUBLISH:
- index.html
- runtime-config.js containing only the public HTTPS backend URL
- vercel.json
- this README

NEVER PUT THESE IN THIS FOLDER:
SARVAM_API_KEY, OLLAMA_API_KEY, GROQ_API_KEY, HF_TOKEN, QDRANT_API_KEY,
GENERATION_API_KEY, OPENCODE_GO_API_KEY, or any .env file.

Before deploying a functional live demo:
1. Deploy the FastAPI backend using DEPLOYMENT.md in the parent project.
2. Open runtime-config.js.
3. Set window.VAANI_API_BASE to the public HTTPS backend URL.
4. Drop this folder into Vercel.

If VAANI_API_BASE is empty, the visual frontend loads but API queries and microphone requests are not connected to a backend. That is safer than putting provider keys in browser code.
