from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Optional, List, Dict, Any
from enum import Enum
from uuid import uuid4

class NodeData(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    nodeId               : str
    className            : str
    text                 : Optional[str] = None
    contentDescription   : Optional[str] = None
    viewIdResourceName   : Optional[str] = None
    isClickable          : bool = False
    isScrollable         : bool = False
    isEditable           : bool = False
    isVisible            : bool = True
    boundsLeft           : int = 0
    boundsTop            : int = 0
    boundsRight          : int = 0
    boundsBottom         : int = 0
    depth                : int = 0
    childCount           : int = 0

    def center_x(self) -> float: return (self.boundsLeft + self.boundsRight) / 2
    def center_y(self) -> float: return (self.boundsTop + self.boundsBottom) / 2
    
    def to_text_repr(self) -> str:
        parts = filter(None, [self.text, self.contentDescription,
                               self.viewIdResourceName, self.className])
        return " ".join(parts)

class ActionType(str, Enum):
    TAP        = "TAP"
    LONG_TAP   = "LONG_TAP"
    TYPE       = "TYPE"
    SCROLL_UP  = "SCROLL_UP"
    SCROLL_DOWN= "SCROLL_DOWN"
    SWIPE      = "SWIPE"
    BACK       = "BACK"
    HOME       = "HOME"

class ActionCommand(BaseModel):
    actionId : str = Field(default_factory=lambda: str(uuid4()))
    nodeId   : str
    type     : ActionType
    text     : Optional[str]  = None
    x        : Optional[float]= None
    y        : Optional[float]= None
    x2       : Optional[float]= None
    y2       : Optional[float]= None

class PlannedStep(BaseModel):
    stepId        : str = Field(default_factory=lambda: str(uuid4()))
    description   : str
    appPackage    : Optional[str] = None
    requiresHitl  : bool = False
    actionType    : ActionType = ActionType.TAP
    inputText     : Optional[str] = None

class TaskPlan(BaseModel):
    taskId  : str = Field(default_factory=lambda: str(uuid4()))
    goal    : str
    steps   : List[PlannedStep]
    context : Dict[str, Any] = {}

# --- WebSocket Message Schemas ---

# INBOUND (Android -> Server)
class ObservationPayload(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    nodes         : List[NodeData]
    activePackage : str
    screenWidth   : int = 1080
    screenHeight  : int = 1920

class AckPayload(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    actionId : str
    status   : str   # "success" | "failed"

class HitlResponsePayload(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    approved : bool

class StartTaskPayload(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    goal   : str
    userId : str = "user_default"

class InboundMessage(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    type    : str
    task_id : str = Field(alias="taskId")
    payload : Dict[str, Any]

# OUTBOUND (Server -> Android)
class CommandPayload(BaseModel):
    action : ActionCommand

class HitlRequestPayload(BaseModel):
    description : str

class StatusPayload(BaseModel):
    status  : str
    message : str = ""

class OutboundMessage(BaseModel):
    type    : str
    taskId  : str
    payload : Dict[str, Any]
