"""job_control.py - lets a user stop whatever the bot is doing for them right now.

Why it was needed: every incoming message is handled while holding a per-user lock, so a long job (sending 100+
papers, summarising a big PDF ...) kept the lock and the user's "/cancel" queued behind it until the job ended.
Now "/cancel" (or /exit, /stop, cancel, exit, stop, quit) is recognised BEFORE the lock: the running job is
cancelled at its next await point (between two messages, downloads or AI calls) and the user's state is cleared.
"""
from __future__ import annotations

import asyncio

CANCEL_WORDS = {"/cancel", "/exit", "/stop", "cancel", "exit", "stop", "quit", "/quit"}

_running: dict[str, asyncio.Task] = {}
_cancelled: set[str] = set()


def is_cancel(text: str) -> bool:
    return (text or "").strip().lower() in CANCEL_WORDS


def register(phone: str) -> None:
    _running[phone] = asyncio.current_task()


def unregister(phone: str) -> None:
    if _running.get(phone) is asyncio.current_task():
        _running.pop(phone, None)


def busy(phone: str) -> bool:
    t = _running.get(phone)
    return bool(t) and not t.done()


def consume_cancel(phone: str) -> bool:
    """True (once) if the CancelledError that just arrived was requested by the user via cancel()."""
    if phone in _cancelled:
        _cancelled.discard(phone)
        return True
    return False


async def cancel(phone: str, wait: float = 5.0) -> bool:
    """Cancels the user's running job. Returns True if there was one."""
    t = _running.get(phone)
    if not t or t.done() or t is asyncio.current_task():
        return False
    _cancelled.add(phone)
    t.cancel()
    await asyncio.wait({t}, timeout=wait)
    _cancelled.discard(phone)
    return True
