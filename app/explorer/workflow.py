"""
Workflow orchestration for the FsExplorer agent.

Written against llama-index-workflows 2.x / llama-index-core 0.12+.

Key API notes:
- Imports from llama_index.core.workflow
- State via ctx.store.edit_state() async context manager
- StopEvent carries payload in result=
- workflow.run() accepts kwargs matching StartEvent fields
"""

import os
from typing import cast, Any

from pydantic import BaseModel

from llama_index.core.workflow import (
    Workflow,
    Context,
    step,
    StartEvent,
    StopEvent,
    Event,
)

from .agent import FsExplorerAgent
from .models import GoDeeperAction, ToolCallAction, StopAction, AskHumanAction, Action
from .fs import describe_dir_content


# =============================================================================
# Workflow State
# =============================================================================

class WorkflowState(BaseModel):
    initial_task: str = ""
    root_directory: str = "."
    current_directory: str = "."


# =============================================================================
# Events
# =============================================================================

class InputEvent(StartEvent):
    task: str
    folder: str = "."


class GoDeeperEvent(Event):
    directory: str
    reason: str


class ToolCallEvent(Event):
    tool_name: str
    tool_input: dict[str, Any]
    reason: str


class ExplorationEndEvent(StopEvent):
    """Terminal event. Payload is stored in StopEvent.result as a dict."""
    pass


# =============================================================================
# Helpers
# =============================================================================

def _make_end(
    final_result: str | None = None,
    error: str | None = None,
) -> ExplorationEndEvent:
    return ExplorationEndEvent(result={"final_result": final_result, "error": error})


async def _run_agent_and_route(
    agent: FsExplorerAgent,
    ctx: Context,
) -> "GoDeeperEvent | ToolCallEvent | ExplorationEndEvent":
    """Call agent.take_action() and convert the result to a workflow event."""
    result = await agent.take_action()

    if result is None:
        return _make_end(error="Agent returned no action.")

    action, action_type = result

    if action_type == "stop":
        stopaction = cast(StopAction, action.action)
        return _make_end(final_result=stopaction.final_result)

    elif action_type == "godeeper":
        godeeper = cast(GoDeeperAction, action.action)
        async with ctx.store.edit_state() as state:
            state.current_directory = godeeper.directory
        event = GoDeeperEvent(directory=godeeper.directory, reason=action.reason)
        ctx.write_event_to_stream(event)
        return event

    elif action_type == "toolcall":
        toolcall = cast(ToolCallAction, action.action)
        event = ToolCallEvent(
            tool_name=toolcall.tool_name,
            tool_input=toolcall.to_fn_args(),
            reason=action.reason,
        )
        ctx.write_event_to_stream(event)
        return event

    else:
        # askhuman not supported in API mode
        return _make_end(error="Agent requested human input (not supported in API mode).")


# =============================================================================
# Workflow
# =============================================================================

class FsExplorerWorkflow(Workflow):
    """
    Event-driven filesystem exploration workflow.
    One fresh instance is created per request inside ExplorerService.
    """

    def __init__(self, agent: FsExplorerAgent, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent = agent

    @step
    async def start_exploration(
        self,
        ctx: Context[WorkflowState],
        ev: InputEvent,
    ) -> "GoDeeperEvent | ToolCallEvent | ExplorationEndEvent":
        root_directory = os.path.abspath(ev.folder)

        if not os.path.exists(root_directory) or not os.path.isdir(root_directory):
            return _make_end(error=f"No such directory: {root_directory}")

        async with ctx.store.edit_state() as state:
            state.initial_task = ev.task
            state.root_directory = root_directory
            state.current_directory = root_directory

        dirdescription = describe_dir_content(root_directory)
        self._agent.configure_task(
            f"The target directory is '{root_directory}' and it contains:\n\n"
            f"```\n{dirdescription}\n```\n\n"
            f"The user is asking: '{ev.task}'\n\n"
            f"Always use absolute file paths when calling tools. "
            f"What action should you take first?"
        )

        return await _run_agent_and_route(self._agent, ctx)

    @step
    async def go_deeper_action(
        self,
        ctx: Context[WorkflowState],
        ev: GoDeeperEvent,
    ) -> "GoDeeperEvent | ToolCallEvent | ExplorationEndEvent":
        state = ctx.store.get_state()
        dirdescription = describe_dir_content(state.current_directory)

        self._agent.configure_task(
            f"Now in directory '{state.current_directory}':\n\n"
            f"```\n{dirdescription}\n```\n\n"
            f"Original task: '{state.initial_task}'\n\n"
            f"What action should you take next?"
        )

        return await _run_agent_and_route(self._agent, ctx)

    @step
    async def tool_call_action(
        self,
        ctx: Context[WorkflowState],
        ev: ToolCallEvent,
    ) -> "GoDeeperEvent | ToolCallEvent | ExplorationEndEvent":
        self._agent.configure_task(
            "Given the result from the tool call you just performed, "
            "what action should you take next?"
        )

        return await _run_agent_and_route(self._agent, ctx)


WORKFLOW_TIMEOUT_SECONDS = 600