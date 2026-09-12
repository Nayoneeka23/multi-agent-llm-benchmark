from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Message(BaseModel):
    sender: str
    receiver: str
    content: str
    round_number: int = 1
    pattern: str = "coordinator_worker"
    network_condition: str = "baseline"
    image_path: Optional[str] = None


class P2PMessage(BaseModel):
    sender: str
    receiver: str
    content: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    round_number: int = 1
    pattern: str = "peer_to_peer"
    network_condition: str = "baseline"
    image_path: Optional[str] = None


class NegotiationMessage(BaseModel):
    sender: str
    receiver: str
    content: str
    round_number: int = 1
    pattern: str = "negotiation"
    network_condition: str = "baseline"
    image_path: Optional[str] = None