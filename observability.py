"""Langfuse tracing, wired via LangChain's own callback integration -- no
hand-rolled span/trace bookkeeping.

Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and optionally
LANGFUSE_HOST, for self-hosted instances) from the environment. If they're
not set, `trace_config()` returns a plain config dict with no callbacks
attached -- the app and eval harness run exactly as before, tracing just
silently doesn't happen, rather than requiring a Langfuse account to run
the app at all.
"""
import os
from typing import Optional

from langfuse.langchain import CallbackHandler

_handler: Optional[CallbackHandler] = None
_warned = False


def get_langfuse_handler() -> Optional[CallbackHandler]:
    """A single shared CallbackHandler, or None if Langfuse isn't configured."""
    global _handler, _warned
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        if not _warned:
            print("[observability] LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set -- tracing disabled.")
            _warned = True
        return None
    if _handler is None:
        _handler = CallbackHandler()
    return _handler


def trace_config(**extra) -> dict:
    """Config dict for a LangChain/LangGraph `.invoke(..., config=...)` call.

    Passing this to any chain or LangGraph app's top-level invoke propagates
    the callback to every nested chain/LLM call in the run (retrieval,
    grading, generation, query rewriting, etc.), so one call site is enough
    to trace an entire request -- no manual span creation needed.
    """
    handler = get_langfuse_handler()
    config = dict(extra)
    if handler:
        config["callbacks"] = [*config.get("callbacks", []), handler]
    return config
