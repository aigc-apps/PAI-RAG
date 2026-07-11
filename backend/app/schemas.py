from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict


class ResponsesRequest(BaseModel):
    # Lenient: the current frontend sends extra fields (enable_agent, kb_ids, ...).
    # Accept and ignore them for now; tools are wired in a later plan.
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    agent_id: Optional[str] = None
    input: Union[str, List[Dict[str, Any]]] = ""
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[str] = None
    user_id: Optional[str] = None
    store: bool = True
    stream: bool = True
    background: bool = False
    metadata: Optional[Dict[str, str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    user: Optional[str] = None
    safety_identifier: Optional[str] = None
    memory: bool = True

    @property
    def resolved_user_id(self) -> Optional[str]:
        return self.user_id or self.user or self.safety_identifier
