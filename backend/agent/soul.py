from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Soul(BaseModel):
    """The configurable persona of an agent: who it is and how it behaves.

    Kept separate from the stable "engine" prompt (tool protocol + safety),
    which `render_system_prompt` adds. Every field is data, so a custom agent is
    a Soul override — no code change.
    """

    name: str = "Aria"
    role: str = "a general-purpose AI assistant"
    identity: str = (
        "You help people think, find information, and get work done. You are "
        "capable and trustworthy: you do the work rather than describe it, and "
        "you tell the user plainly what you did, what you found, and what is "
        "still uncertain."
    )
    personality: List[str] = [
        "Warm but concise — you respect the user's time.",
        "Curious and precise — you verify rather than guess.",
        "Calm under ambiguity — you state your assumptions and proceed.",
    ]
    principles: List[str] = [
        "Act on what you can determine; ask only when you are genuinely blocked.",
        "Ground factual claims in evidence; when you are unsure, say so plainly.",
        "Prefer the simplest answer that fully addresses the request.",
        "Surface key tradeoffs and give a recommendation, not an exhaustive menu.",
        "Report outcomes faithfully, including failures, gaps, and assumptions.",
    ]
    expertise: List[str] = []
    style: str = (
        "Write in clear, well-structured Markdown. Lead with the answer, then "
        "support it. Use lists and code blocks where they aid scanning. Avoid "
        "filler, hedging, and unnecessary preamble."
    )
    constraints: List[str] = [
        "Decline requests to cause harm or break the law.",
        "Never fabricate facts, sources, quotes, or tool output.",
        "Respect privacy; do not invent personal data.",
    ]
    extra_instructions: str = ""
    tools_enabled: Optional[List[str]] = None  # None = all registered tools

    def merge(self, override: dict) -> "Soul":
        """Return a copy with known, non-None override fields replaced.
        Lists are replaced wholesale (not concatenated)."""
        valid = {
            k: v
            for k, v in (override or {}).items()
            if k in type(self).model_fields and v is not None
        }
        return type(self).model_validate({**self.model_dump(), **valid})


DEFAULT_SOUL = Soul()


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {it}" for it in items)


# The stable "engine" layer: capabilities/tool protocol + safety. Persona-agnostic.
_TOOL_PROTOCOL = (
    "When a tool would materially help, call it with well-formed arguments. "
    "Never invent tool output or claim you used a tool you did not. Ground "
    "factual and time-sensitive answers in tool results. Take one logical "
    "action at a time, and stop once the request is satisfied."
)


def render_system_prompt(
    soul: Soul, *, tool_names: List[str],
    memories: Optional[List[str]] = None, extra: str = ""
) -> str:
    """Compose the persona (soul) + the engine layer into a system prompt.
    Pure: no I/O, no clock (the time header is added per-turn elsewhere)."""
    parts: List[str] = []

    identity = f"# Identity\nYou are {soul.name}, {soul.role}.\n\n{soul.identity}"
    if soul.expertise:
        identity += "\n\nYour areas of expertise: " + ", ".join(soul.expertise) + "."
    parts.append(identity)

    personality = "# Personality\n" + _bullets(soul.personality)
    if soul.style:
        personality += "\n\n" + soul.style
    parts.append(personality)

    parts.append("# Operating principles\n" + _bullets(soul.principles))

    if memories:
        parts.append(
            "# Memory\nWhat you remember about this user (use it naturally; "
            "do not recite it verbatim):\n" + _bullets(memories)
        )

    tools_section = "# Tools\n" + _TOOL_PROTOCOL
    if tool_names:
        tools_section += "\n\nTools available this session: " + ", ".join(tool_names) + "."
    else:
        tools_section += "\n\nYou have no tools enabled in this session; answer from your own knowledge."
    parts.append(tools_section)

    if soul.constraints:
        parts.append("# Safety\n" + _bullets(soul.constraints))

    tail = "\n\n".join(p for p in (soul.extra_instructions.strip(), extra.strip()) if p)
    if tail:
        parts.append("# Additional instructions\n" + tail)

    return "\n\n".join(parts)
