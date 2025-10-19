from typing import List, Dict, Any


async def generate_chat(system: str, messages: List[Dict[str, Any]], timeout_s: int = 20) -> str:
    """Placeholder LLM call.

    Tests monkeypatch this function, so the implementation here only serves
    as a safe default if not mocked.
    """
    return ""

