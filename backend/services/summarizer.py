from typing import List, Dict, Any, Optional


async def summarize_messages(messages: List[Dict[str, Any]], previous_summary: Optional[str] = None) -> str:
    """Return a trivial summary.

    Tests monkeypatch this function. This default keeps behavior deterministic
    if not monkeypatched.
    """
    return previous_summary or ""

