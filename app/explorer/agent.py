"""
FsExplorer Agent — filesystem exploration using Google Gemini.

MVP 1: core agentic loop with page-range-aware system prompt.
No index, no graph. Pure agentic reasoning that works on any file size.
"""

import os
from typing import Callable, Any, cast
from dataclasses import dataclass

from google.genai.types import Content, HttpOptions, Part
from google.genai import Client as GenAIClient

from .models import Action, ActionType, ToolCallAction, Tools
from .fs import (
    read_file,
    grep_file_content,
    glob_paths,
    scan_folder,
    preview_file,
    parse_file,
)
from app.utils.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# Token Usage Tracking
# =============================================================================

GEMINI_FLASH_INPUT_COST_PER_MILLION = 0.075
GEMINI_FLASH_OUTPUT_COST_PER_MILLION = 0.30


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    tool_result_chars: int = 0
    documents_parsed: int = 0
    documents_scanned: int = 0

    def add_api_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.api_calls += 1

    def add_tool_result(self, result: str, tool_name: str) -> None:
        self.tool_result_chars += len(result)
        if tool_name in ("parse_file", "preview_file"):
            self.documents_parsed += 1
        elif tool_name == "scan_folder":
            self.documents_scanned += result.count("│ [")

    def _calculate_cost(self) -> tuple[float, float, float]:
        input_cost = (
            self.prompt_tokens / 1_000_000
        ) * GEMINI_FLASH_INPUT_COST_PER_MILLION
        output_cost = (
            self.completion_tokens / 1_000_000
        ) * GEMINI_FLASH_OUTPUT_COST_PER_MILLION
        return input_cost, output_cost, input_cost + output_cost

    def to_dict(self) -> dict:
        input_cost, output_cost, total_cost = self._calculate_cost()
        return {
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "documents_scanned": self.documents_scanned,
            "documents_parsed": self.documents_parsed,
            "tool_result_chars": self.tool_result_chars,
            "estimated_cost_usd": round(total_cost, 6),
        }


# =============================================================================
# Tool Registry
# =============================================================================

TOOLS: dict[Tools, Callable[..., str]] = {
    "read": read_file,
    "grep": grep_file_content,
    "glob": glob_paths,
    "scan_folder": scan_folder,
    "preview_file": preview_file,
    "parse_file": parse_file,
}


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are a document research agent. You explore files to answer user questions accurately, with citations.

## Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `scan_folder` | Preview ALL documents in a folder at once | `directory` |
| `preview_file` | Quick look at one document (first 10 pages) | `file_path` |
| `parse_file` | Read document content, full or by page range | `file_path`, `page_start`, `page_end` |
| `read` | Read a plain text file | `file_path` |
| `grep` | Search for a pattern inside a file | `file_path`, `pattern` |
| `glob` | Find files matching a pattern | `directory`, `pattern` |

## Strategy

### Step 1 — Scan first
Always start with `scan_folder` to see what files exist and get a preview of each.
In your reason, categorize every file: RELEVANT / MAYBE / SKIP.

### Step 2 — Read in sections for large documents
For any document over ~50 pages, NEVER parse the full document at once.
Read it in sections of 50 pages using page_start and page_end:

  First section:  parse_file(path, page_end=50)
  Next section:   parse_file(path, page_start=51, page_end=100)
  Next section:   parse_file(path, page_start=101, page_end=150)
  ... and so on.

STOP reading as soon as you have enough information to answer.
You do NOT need to read every section if you find the answer earlier.

### Step 3 — Follow cross-references
If a section says "see Article X" or "refer to Schedule B":
- Note the reference in your reason
- Use grep or parse_file on the relevant page range to find that section
- This is how you resolve cross-document and cross-section dependencies

### Step 4 — Cite everything
Every factual claim in your final answer needs a citation.
Format: [Source: filename, page range or section]

End with:
## Sources Consulted
- filename — what it contained / which sections were read

## Important Rules
- Always use absolute file paths when calling tools
- For large documents, page ranges are mandatory — never load 100+ pages at once
- Stop exploring as soon as you have a complete answer
- If you find a cross-reference, follow it before concluding
"""


# =============================================================================
# Agent
# =============================================================================


class FsExplorerAgent:
    """
    AI agent for filesystem exploration using Google Gemini.
    Maintains conversation history across tool calls within a single query.
    One instance per request — stateless between requests.
    """

    def __init__(self, api_key: str | None = None) -> None:
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY not found. Add it to your .env file.")
        self._client = GenAIClient(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1beta"),
        )
        self._chat_history: list[Content] = []
        self.token_usage = TokenUsage()

    def configure_task(self, task: str) -> None:
        """Append a user message to the conversation history."""
        self._chat_history.append(
            Content(role="user", parts=[Part.from_text(text=task)])
        )

    async def take_action(self) -> tuple[Action, ActionType] | None:
        """Send chat history to Gemini and get the next structured action."""
        log.info(
            f"API call #{self.token_usage.api_calls + 1} — history length: {len(self._chat_history)}"
        )

        response = await self._client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=self._chat_history,  # type: ignore
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": Action,
            },
        )

        if response.usage_metadata:
            self.token_usage.add_api_call(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
            )
            log.info(
                f"Tokens used: {response.usage_metadata.prompt_token_count} in / "
                f"{response.usage_metadata.candidates_token_count} out"
            )

        if response.candidates is not None:
            if response.candidates[0].content is not None:
                self._chat_history.append(response.candidates[0].content)
            if response.text is not None:
                action = Action.model_validate_json(response.text)
                action_type = action.to_action_type()
                log.info(f"Agent action: {action_type} — {action.reason[:80]}...")

                if action_type == "toolcall":
                    toolcall = cast(ToolCallAction, action.action)
                    self.call_tool(
                        tool_name=toolcall.tool_name,
                        tool_input=toolcall.to_fn_args(),
                    )
                return action, action_type

        return None

    def call_tool(self, tool_name: Tools, tool_input: dict[str, Any]) -> None:
        """Execute a tool and append the result to conversation history."""
        log.info(f"Tool call: {tool_name}({tool_input})")
        try:
            result = TOOLS[tool_name](**tool_input)
        except Exception as e:
            result = f"Error calling tool '{tool_name}' with args {tool_input}: {e}"
            log.error(result)

        self.token_usage.add_tool_result(result, tool_name)
        log.info(f"Tool result: {len(result):,} chars")

        self._chat_history.append(
            Content(
                role="user",
                parts=[
                    Part.from_text(text=f"Tool result for {tool_name}:\n\n{result}")
                ],
            )
        )

    def reset(self) -> None:
        """Reset conversation history and token tracking."""
        self._chat_history.clear()
        self.token_usage = TokenUsage()
