// Public backend URL. Never put provider secrets in this file.
window.VAANI_API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? ""
  : "https://vaani-rag-api.onrender.com";

