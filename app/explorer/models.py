"""
Pydantic models for FsExplorer agent actions.
"""

from pydantic import BaseModel, Field
from typing import TypeAlias, Literal, Any


# =============================================================================
# Type Aliases
# =============================================================================

Tools: TypeAlias = Literal[
    "read",
    "grep",
    "glob",
    "scan_folder",
    "preview_file",
    "parse_file",
]

ActionType: TypeAlias = Literal["stop", "godeeper", "toolcall", "askhuman"]


# =============================================================================
# Action Models
# =============================================================================

class StopAction(BaseModel):
    final_result: str = Field(
        description="Final result of the operation with the answer to the user's query"
    )


class AskHumanAction(BaseModel):
    question: str = Field(
        description="Clarification question to ask the user"
    )


class GoDeeperAction(BaseModel):
    directory: str = Field(
        description="Path to the directory to navigate into"
    )


class ToolCallArg(BaseModel):
    parameter_name: str = Field(description="Name of the parameter")
    parameter_value: Any = Field(description="Value for the parameter")


class ToolCallAction(BaseModel):
    tool_name: Tools = Field(description="Name of the tool to invoke")
    tool_input: list[ToolCallArg] = Field(description="Arguments to pass to the tool")

    def to_fn_args(self) -> dict[str, Any]:
        return {arg.parameter_name: arg.parameter_value for arg in self.tool_input}


class Action(BaseModel):
    action: ToolCallAction | GoDeeperAction | StopAction | AskHumanAction = Field(
        description="The specific action to take"
    )
    reason: str = Field(description="Explanation for why this action was chosen")

    def to_action_type(self) -> ActionType:
        if isinstance(self.action, ToolCallAction):
            return "toolcall"
        elif isinstance(self.action, GoDeeperAction):
            return "godeeper"
        elif isinstance(self.action, AskHumanAction):
            return "askhuman"
        else:
            return "stop"