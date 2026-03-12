"""
API router for document exploration endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.explorer_service import ExplorerService

router = APIRouter(prefix="/api", tags=["explorer"])

_service = ExplorerService()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str | None
    error: str | None
    usage: dict


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    """
    Run an agentic document search against the DATA folder.

    - **query**: The question to answer based on documents in DATA/
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = await _service.query(task=payload.query)

    return QueryResponse(
        answer=result["answer"],
        error=result["error"],
        usage=result["usage"],
    )