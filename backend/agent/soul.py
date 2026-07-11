from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Soul(BaseModel):
    """The configurable persona of an agent: who it is and how it behaves.

    Kept separate from the stable "engine" prompt (tool protocol + safety),
    which `render_stable_system_prompt` adds. Every field is data, so a custom agent is
    a Soul override — no code change.
    """

    name: str = "MiniAgent"
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


# The stable "engine" layer: tool protocol + execution bias. Persona-agnostic and
# always on, so persistence and verification survive any Soul persona override
# (they deliberately do NOT live in soul.principles, which a custom agent replaces).
_TOOL_PROTOCOL = (
    "When a tool would materially help, call it with well-formed arguments. "
    "Never invent tool output or claim you used a tool you did not. Ground "
    "factual and time-sensitive answers in tool results. Work in deliberate "
    "steps and keep going until the request is fully handled — don't stop at a "
    "partial result or hand back a plan you could have carried out yourself. If a "
    "tool returns nothing useful or fails, adjust and try another angle before "
    "giving up or falling back on guesswork. Before finalizing, check that what "
    "you produced actually answers what was asked; then stop rather than "
    "over-working."
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


_ALIYUN_CLI_GUIDANCE = (
    "This user has connected an Alibaba Cloud (阿里云) account. You can run the "
    "`aliyun` CLI in the sandbox (via the shell tool) to view or operate their "
    "cloud resources — PAI, EAS, and related services. Their cloud identity is "
    "injected into the sandbox automatically (temporary STS credentials, via the "
    "environment and a pre-configured CLI profile), so run `aliyun ...` commands "
    "directly. Do NOT run `aliyun configure`, edit ~/.aliyun/config.json, or set "
    "access keys yourself — credentials are managed for you and any manual setup "
    "will be wrong. If an aliyun command fails with a credential or authorization "
    "error, authorization is a one-click action the user performs in their own "
    "cloud account (the UI shows them a card) — you cannot do it for them, so stop "
    "and let them authorize rather than retrying or trying to configure the CLI. "
    "Do NOT discard stderr when running aliyun (avoid `2>/dev/null` and the like): "
    "the error text is what identifies an authorization problem, so keep it visible."
)


# Added when the knowledge subsystem is wired in (anchored on knowledge_search).
# Reflex-level: ground answers in ingested docs, and disambiguate the four KB tools
# so the model picks the right one instead of defaulting to knowledge_search for
# everything. Heavier procedures (citation discipline, cross-KB compare) live in the
# knowledge_qa skill, not here.
_KNOWLEDGE_GUIDANCE = (
    "A knowledge base of ingested documents is available. Before answering a "
    "question its contents could cover, search it and ground your answer in what "
    "you find, citing the source; if it does not contain the answer, say so plainly "
    "rather than guessing. Pick the right tool: knowledge_search for a "
    "meaning/keyword query (the default; it searches every accessible base at once "
    "unless you pass kb_ids); grep_file for an exact literal string that tokenized "
    "search misses — error codes, identifiers, API names, exact jargon; view_file to "
    "read a whole document once a hit looks relevant, or view_file(chunk_id=…, "
    "mode=\"locate\") to open a passage in its surrounding context; "
    "list_knowledge_bases to see which bases exist and get their ids when you need to "
    "narrow a search. Prefer these over your own recall for anything the docs cover."
)


# Added when the execution sandbox is wired in (code_interpreter and/or shell).
# Reflex-level: reach for execution instead of computing in your head, and split the
# two entrypoints. publish_artifact's own block covers surfacing files, so this does
# not repeat it.
_SANDBOX_GUIDANCE = (
    "You have a sandbox that runs real code and shell commands. Reach for it "
    "whenever execution beats reasoning: non-trivial arithmetic, parsing or "
    "transforming files, data analysis, running or testing a script, checking an "
    "actual command's output — do not compute large or precise results in your head. "
    "Use code_interpreter to run code (e.g. Python) and shell to run shell commands; "
    "they share the same environment and mounts, including the durable per-user "
    "directory at /mnt/user ($AGENT_USER_PATH). The shell runs operating-system and "
    "CLI commands only — it is not a way to invoke your own tools: load_skill, "
    "knowledge_search, publish_artifact and the rest are tool/function calls, so call "
    "them directly and never type a tool name as a shell command. Keep stderr visible "
    "when a command fails so you can see why, and fix and retry rather than guessing "
    "at the result."
)


# Added when a read-only code layer is baked in at /opt/code (gated on both the
# layer being configured and a sandbox tool being present to explore it). The
# knowledge base is the primary ground truth; the code is the fallback when it
# comes up empty on questions about the system's own implementation.
_CODE_LAYER_GUIDANCE = (
    "A read-only code layer is available at /opt/code ($AGENT_CODE_PATH), holding "
    "the source repositories behind this system, one per subdirectory. When "
    "knowledge_search / the knowledge base does not answer a question that is "
    "really about how this system's code behaves, fall back to the code: run "
    "`ls /opt/code` to see which repositories are available, then explore the "
    "relevant one with shell / code_interpreter (ripgrep or grep to find "
    "symbols, cat to read files). It is read-only reference material — do not "
    "try to modify it — and it is a fallback for source-level questions, not a "
    "replacement for knowledge_search on document questions."
)


def _code_layer_block(code_manifest: str) -> str:
    """The /opt/code guidance. With a manifest (the per-agent, admin-curated
    list of what each repo is), lead with it so the model knows the repos up
    front; without one, fall back to discover-by-`ls`."""
    manifest = (code_manifest or "").strip()
    if not manifest:
        return _CODE_LAYER_GUIDANCE
    return (
        "A read-only code layer is available at /opt/code ($AGENT_CODE_PATH), "
        "holding the source repositories behind this system, one per "
        "subdirectory. The available repositories:\n\n" + manifest + "\n\n"
        "When knowledge_search / the knowledge base does not answer a question "
        "that is really about how this system's code behaves, fall back to the "
        "code: open the relevant repository under /opt/code and explore it with "
        "shell / code_interpreter (ripgrep or grep to find symbols, cat to read "
        "files); run `ls /opt/code` for anything the list above does not cover. "
        "It is read-only reference material — do not try to modify it — and it "
        "is a fallback for source-level questions, not a replacement for "
        "knowledge_search on document questions."
    )


def render_stable_system_prompt(
    soul: Soul, *, tool_names: List[str], project_context: str = "",
    aliyun_pai_enabled: bool = False, code_layer_enabled: bool = False,
    code_manifest: str = "",
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
        # Subsystem guidance, each gated on its anchor tool being present. Short and
        # always-on so a reflex capability never depends on the model loading a skill
        # first; verbose workflows still live in skills.
        if "knowledge_search" in tool_names:
            tools_section += "\n\n" + _KNOWLEDGE_GUIDANCE
        if "code_interpreter" in tool_names or "shell" in tool_names:
            tools_section += "\n\n" + _SANDBOX_GUIDANCE
            if code_layer_enabled:
                tools_section += "\n\n" + _code_layer_block(code_manifest)
        if "publish_artifact" in tool_names:
            tools_section += "\n\n" + _FILE_OUTPUT_GUIDANCE
        if aliyun_pai_enabled and "shell" in tool_names:
            tools_section += "\n\n" + _ALIYUN_CLI_GUIDANCE
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
