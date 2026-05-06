from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from typing import List, Optional, Any
import uuid

# Base model to handle camelCase <-> snake_case automatically
class AetherModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True
    )

# --- Action Models ---
class NodeData(AetherModel):
    node_id: str
    class_name: str
    text: Optional[str] = None
    content_description: Optional[str] = None
    view_id_resource_name: Optional[str] = None
    is_clickable: bool = False
    is_scrollable: bool = False
    is_editable: bool = False
    is_visible: bool = True
    bounds_left: int = 0
    bounds_top: int = 0
    bounds_right: int = 0
    bounds_bottom: int = 0
    depth: int = 0
    child_count: int = 0

    def to_text_repr(self) -> str:
        parts = [f"[{self.class_name}]"]
        if self.text: parts.append(f"text='{self.text}'")
        if self.content_description: parts.append(f"desc='{self.content_description}'")
        if self.view_id_resource_name: parts.append(f"id='{self.view_id_resource_name}'")
        parts.append(f"at({self.bounds_left},{self.bounds_top})")
        return " ".join(parts)

class ActionCommand(AetherModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str # tap, long_tap, scroll_up, scroll_down, type, swipe, back, home
    node_id: Optional[str] = None
    text: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None

class TaskPlan(AetherModel):
    task_id: str
    goal: str
    steps: List[str] = []
    status: str = "pending"
    context: dict = {}

# --- WebSocket Inbound (from Android) ---
class StartTaskPayload(AetherModel):
    goal: str
    user_id: str

class ObservationPayload(AetherModel):
    nodes: List[NodeData]
    active_package: str
    screen_width: int
    screen_height: int

class AckPayload(AetherModel):
    action_id: str
    status: str

class HitlResponsePayload(AetherModel):
    approved: bool

class InboundMessage(AetherModel):
    type: str  # start_task, observation, ack, hitl_response
    task_id: str
    payload: Any

# --- WebSocket Outbound (to Android) ---
class CommandPayload(AetherModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    action: ActionCommand
    thought: str

class StatusPayload(AetherModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    message: str
    status: str

class HitlRequestPayload(AetherModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    description: str

class OutboundMessage(AetherModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    type: str # command, status, task_complete, task_failed, hitl_request
    task_id: str
    payload: Any
