"""
ExplorerService — runs one query against the DATA folder.

Creates a fresh agent + workflow per request (no shared state between calls).
"""

from app.config.settings import GOOGLE_API_KEY, DATA_DIR
from app.explorer.agent import FsExplorerAgent
from app.explorer.workflow import (
    FsExplorerWorkflow,
    InputEvent,
    WORKFLOW_TIMEOUT_SECONDS,
)
from app.utils.logging import get_logger

log = get_logger(__name__)


class ExplorerService:
    async def query(self, task: str) -> dict:
        agent = FsExplorerAgent(api_key=GOOGLE_API_KEY)

        workflow = FsExplorerWorkflow(
            agent=agent,
            timeout=WORKFLOW_TIMEOUT_SECONDS,
        )

        handler = workflow.run(start_event=InputEvent(task=task, folder=DATA_DIR))

        # llama-index-workflows 2.x: awaiting the handler returns whatever
        # was passed into StopEvent(result=...).
        # Depending on the exact patch version, this may be:
        #   (a) the dict directly: {"final_result": ..., "error": ...}
        #   (b) the StopEvent object, with .result holding that dict
        raw = await handler

        log.info(
            f"Workflow raw result type: {type(raw).__name__}, value: {repr(raw)[:200]}"
        )

        # Unwrap safely regardless of which form we get back
        if isinstance(raw, dict):
            payload = raw
        elif hasattr(raw, "result") and isinstance(raw.result, dict):
            payload = raw.result
        elif hasattr(raw, "result") and raw.result is not None:
            payload = {"final_result": str(raw.result), "error": None}
        else:
            # Last resort: stringify whatever came back
            payload = {"final_result": str(raw) if raw else None, "error": None}

        answer = payload.get("final_result")
        error = payload.get("error")

        log.info(
            f"Final answer length: {len(answer) if answer else 0} chars, error: {error}"
        )

        return {
            "answer": answer,
            "error": error,
            "usage": agent.token_usage.to_dict(),
        }
