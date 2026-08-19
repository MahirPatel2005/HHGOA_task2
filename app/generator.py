import os
import time
from typing import Any
from app.providers import (
    AnthropicAnswerGenerator,
    OpenAICompatibleAnswerGenerator,
    OpenCodeGoResponsesAnswerGenerator,
    OllamaCloudAnswerGenerator,
    TemplateAnswerGenerator,
    get_opencode_go_api_key,
)

class AnswerObject:
    def __init__(self, text: str, grounded: bool, generation_ms: float, model: str):
        self.text = text
        self.grounded = grounded
        self.generation_ms = generation_ms
        self.model = model

_generator = None

def get_generator():
    global _generator
    if _generator is not None:
        return _generator
        
    generation_key = os.getenv("GROQ_API_KEY") or os.getenv("GENERATION_API_KEY")
    ollama_key = os.getenv("OLLAMA_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    generation_provider = os.getenv("GENERATION_PROVIDER", "auto").lower()
    opencode_go_key = get_opencode_go_api_key()
    
    if generation_provider in {"ollama", "ollama-cloud", "ollama_cloud"} and ollama_key:
        _generator = OllamaCloudAnswerGenerator(
            api_key=ollama_key,
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
            timeout_s=float(os.getenv("GENERATION_TIMEOUT_S", "30")),
        )
    elif generation_provider == "anthropic" and anthropic_key:
        _generator = AnthropicAnswerGenerator(
            api_key=anthropic_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            timeout_s=float(os.getenv("GENERATION_TIMEOUT_S", "20")),
        )
    elif generation_provider in {"opencode-go", "opencode_go"} and opencode_go_key:
        _generator = OpenCodeGoResponsesAnswerGenerator(
            api_key=opencode_go_key,
            model=os.getenv("OPENCODE_GO_MODEL", "gpt-5.6-luna"),
            base_url=os.getenv("OPENCODE_GO_BASE_URL"),
        )
    elif generation_provider == "auto" and opencode_go_key and not generation_key and not anthropic_key:
        _generator = OpenCodeGoResponsesAnswerGenerator(
            api_key=opencode_go_key,
            model=os.getenv("OPENCODE_GO_MODEL", "gpt-5.6-luna"),
            base_url=os.getenv("OPENCODE_GO_BASE_URL"),
        )
    elif generation_provider == "auto" and anthropic_key:
        _generator = AnthropicAnswerGenerator(
            api_key=anthropic_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            timeout_s=float(os.getenv("GENERATION_TIMEOUT_S", "20")),
        )
    elif generation_key:
        _generator = OpenAICompatibleAnswerGenerator(
            api_key=generation_key,
            model=os.getenv("GENERATION_MODEL", "llama-3.1-8b-instant"),
            base_url=os.getenv("GENERATION_BASE_URL", "https://api.groq.com/openai/v1"),
        )
    else:
        _generator = TemplateAnswerGenerator()
        if hasattr(_generator, "_init_query_sets"):
            _generator._init_query_sets()
    return _generator

def generate_answer(query: str, results: list[Any]) -> AnswerObject:
    t0 = time.perf_counter()
    generator = get_generator()
    contexts = [r.text for r in results]
    try:
        ans_text = generator.generate(query, contexts)
        model = getattr(generator, "model", "local-generator")
    except Exception as exc:
        fallback = TemplateAnswerGenerator()
        ans_text = fallback.generate(query, contexts)
        model = f"fallback-template-generator (due to error: {type(exc).__name__})"
    grounded = "I don't have enough evidence" not in ans_text
    generation_ms = (time.perf_counter() - t0) * 1000
    return AnswerObject(
        text=ans_text,
        grounded=grounded,
        generation_ms=generation_ms,
        model=model,
    )

# Pre-initialize generator and query sets on module import
try:
    get_generator()
except Exception:
    pass
