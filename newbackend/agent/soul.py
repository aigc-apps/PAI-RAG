from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Soul(BaseModel):
    """The configurable persona of an agent: who it is and how it behaves.

    Kept separate from the stable "engine" prompt (tool protocol + safety),
    which `render_stable_system_prompt` adds. Every field is data, so a custom agent is
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

# Only added when publish_artifact is registered. Without it, models tend to
# write a file in the sandbox and then merely describe it or print its path,
# which the user cannot open. This tells the model to hand the file to the UI.
_FILE_OUTPUT_GUIDANCE = (
    "When you create a file the user should see or keep — a report, chart, "
    "image, HTML page, diagram, or data export — save it under /mnt/user "
    "(the durable per-user directory, also available as $AGENT_USER_PATH in the "
    "sandbox) and then call publish_artifact with its path. That surfaces the "
    "file in the UI: markdown, images, and HTML preview in a side panel, other "
    "types get a download link. Do not just print the sandbox path or offer to "
    "paste the file's contents — publish it so the user can actually open it. "
    "The sandbox has no network the user's browser can reach, so never start a "
    "web server (e.g. python -m http.server) or point the user at a localhost "
    "URL; publish_artifact is the only way to surface a file."
)


def render_stable_system_prompt(
    soul: Soul, *, tool_names: List[str], project_context: str = ""
) -> str:
    """Stable, cacheable layer: persona + project + tool protocol + safety.
    Excludes volatile content (memory, per-request instructions, conversation summary)."""
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
    if project_context.strip():
        parts.append("# Project context\n" + project_context.strip())
    tools_section = "# Tools\n" + _TOOL_PROTOCOL
    if tool_names:
        tools_section += "\n\nTools available this session: " + ", ".join(tool_names) + "."
        if "publish_artifact" in tool_names:
            tools_section += "\n\n" + _FILE_OUTPUT_GUIDANCE
    else:
        tools_section += "\n\nYou have no tools enabled in this session; answer from your own knowledge."
    parts.append(tools_section)
    if soul.constraints:
        parts.append("# Safety\n" + _bullets(soul.constraints))
    return "\n\n".join(parts)


def render_context_block(
    *, memories: Optional[List[str]] = None, summary: str = "", instructions: str = ""
) -> str:
    """Volatile per-turn context: user memory + rolling conversation summary +
    per-request instructions. Returns '' when all are empty."""
    parts: List[str] = []
    if memories:
        parts.append(
            "# Memory\nWhat you remember about this user (use it naturally; "
            "do not recite it verbatim):\n" + _bullets(memories)
        )
    if summary.strip():
        parts.append(
            "# Conversation summary\nSummary of earlier turns in this conversation:\n"
            + summary.strip()
        )
    if instructions.strip():
        parts.append("# Additional instructions\n" + instructions.strip())
    return "\n\n".join(parts)
