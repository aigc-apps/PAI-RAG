from __future__ import annotations
from typing import List
from loguru import logger
from agent.context import AgentContext, Attachment, RunVars
from agent.message import Message


def _format_attachments(attachments: List[Attachment]) -> str:
    if not attachments:
        return ""
    blocks = [f'<attached_file name="{a.name}">\n{a.body}\n</attached_file>' for a in attachments]
    return "\n\n以下是用户本次上传的文件内容，请直接基于这些内容回答：\n\n" + "\n\n".join(blocks)


def render_current_turn(turn: Message, attachments: List[Attachment],
                        hints: List[str], run_vars: RunVars) -> Message:
    """Assemble the live user turn: time header + user text + attachment blocks + hints.
    The ONLY place these are combined. Handles both str and multimodal-list content."""
    prefix = f"[System Time: {run_vars.current_datetime}]\n"
    suffix = _format_attachments(attachments)
    if hints:
        suffix += "\n\n" + "\n\n".join(hints)

    if isinstance(turn.content, list):
        parts = [dict(p) for p in turn.content]
        for p in parts:
            if p.get("type") == "text":
                p["text"] = prefix + (p.get("text") or "") + suffix
                break
        else:
            parts.insert(0, {"type": "text", "text": prefix + suffix})
        return Message(role="user", content=parts)

    base = turn.content or ""
    return Message(role="user", content=prefix + base + suffix)


class Agent:
    @staticmethod
    def build_messages(ctx: AgentContext) -> List[Message]:
        msgs: List[Message] = [Message("system", ctx.system_prompt)]
        msgs += ctx.history
        msgs.append(render_current_turn(ctx.current_turn, ctx.attachments, ctx.hints, ctx.run_vars))
        logger.info(
            "[agent] model input: %d msgs; current turn head=%r",
            len(msgs),
            (msgs[-1].content if isinstance(msgs[-1].content, str) else "<multimodal>")[:200],
        )
        return msgs
