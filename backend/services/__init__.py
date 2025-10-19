# Mark services as a package and expose key service modules for tests to monkeypatch.

from . import gemini as gemini  # noqa: F401
from . import summarizer as summarizer  # noqa: F401

__all__ = [
    "gemini",
    "summarizer",
]

