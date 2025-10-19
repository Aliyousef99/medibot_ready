from enum import Enum
from pydantic import BaseModel

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class Message(BaseModel):
    role: MessageRole
    content: str
