import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the backend/ root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ── Google Gemini ────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

# ── Data directory ───────────────────────────────────────────────────────────
# Resolves to  backend/DATA/  regardless of where uvicorn is launched from.
DATA_DIR: str = str(Path(__file__).parent.parent.parent / "DATA")
