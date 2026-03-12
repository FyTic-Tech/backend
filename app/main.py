from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import router

app = FastAPI(
    title="Agentic File Explorer",
    description="AI-powered document search using agentic filesystem exploration.",
    version="1.0.0",
)

# Allow all origins for local development.
# Tighten this when you have a real frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
