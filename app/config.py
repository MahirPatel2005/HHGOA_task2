import os
from dotenv import load_dotenv

# Automatically load .env environment variables when imported by eval runner
load_dotenv()

GENERATION_BACKEND = os.getenv("GENERATION_PROVIDER", "anthropic")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
LATENCY_BUDGET_MS = 200.0
