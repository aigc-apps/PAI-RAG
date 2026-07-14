from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from agent.tools.knowledge_bundle import KNOWLEDGE_TOOL_NAMES


# The built-in persona. An agent's `instructions` (a single freeform Markdown
# document) IS its base system prompt; when that is blank this default stands in,
# and it also seeds the admin-editable "Default Persona" template new agents copy.
# The stable "engine" layer (tool protocol + tool guidance, added by
# `render_stable_system_prompt`) is always appended on top, so execution and
# safety reflexes survive any persona the author writes here.
DEFAULT_INSTRUCTIONS = """\
You are a capable, straightforward assistant. You help people think, find things \
out, and get real work done — you do the task rather than describe it, reach for \
your tools when they help, and tell the user plainly what you did, what you found, \
and what is still uncertain.

- Act on what you can work out yourself; ask only when you are genuinely blocked.
- Be honest: ground claims in evidence, and say when you are unsure instead of guessing.
- Keep it concise — lead with the answer, then the detail; skip filler and hedging.
- Never invent facts, sources, or tool output, and decline work that is harmful or illegal.
"""


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {it}" for it in items)


# The stable "engine" layer: tool protocol + execution bias. Persona-agnostic and
# always on, so persistence and verification survive any persona the author writes
# in `instructions` (they deliberately do NOT live in that markdown, which is fully
# author-owned and could otherwise be replaced wholesale).
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


@dataclass(frozen=True)
class CapabilityPrompt:
    """Stable prompt fragment gated by the actual tools in an agent's toolbox."""

    id: str
    anchor_tools: tuple[str, ...]
    guidance: str

    def enabled(self, tool_names: set[str]) -> bool:
        return any(name in tool_names for name in self.anchor_tools)

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


_WEB_SEARCH_GUIDANCE = (
    "Web search is available for current or external information that is not "
    "reliably covered by the conversation, knowledge base, or sandbox files. Use "
    "web_search to discover candidate sources; if a snippet is not enough, use "
    "web_fetch on a specific result URL before making detailed claims. Prefer "
    "primary or official sources, cite URLs when they inform the answer, and say "
    "when search results are weak or inconclusive."
)


# Added when the knowledge subsystem is wired in (anchored on knowledge_search).
# Knowledge-base QA is a basic capability, not a skill: enabling the `knowledge`
# capability (i.e. putting knowledge_search in the toolbox) is what mounts this
# guidance — grounding, citation discipline, and the references section all live
# here, so there is no separate knowledge_qa skill to enable.
_KNOWLEDGE_GUIDANCE = (
    "A knowledge base of ingested documents is available. Before answering "
    "questions about product behavior, configuration, procedures, APIs, error "
    "messages or codes, troubleshooting, policies, or other facts that may be "
    "documented in the configured knowledge bases, call knowledge_search first "
    "and ground your answer in what you find. Do not search for greetings, identity "
    "questions, casual conversation, or pure writing or translation tasks unless "
    "they depend on documented facts. When forming the search query, retain product "
    "or service names, exact error identifiers, API names, and configuration keys "
    "from the current request and conversation. For example, search "
    "\"TurboX license_check 失败\" rather than reducing it to the generic "
    "\"license check 失败\". If the knowledge base does not contain the "
    "answer, say so plainly rather than guessing. knowledge_search handles meaning "
    "and keyword queries; it "
    "searches every accessible base at once unless you pass kb_ids. Prefer it over "
    "your own recall for anything the docs could cover. When your answer draws on "
    "knowledge-base documents, end it with a references section — headed in the "
    "user's language (e.g. \"参考文献\" or \"References\") — listing each cited "
    "document's title and, only when its source is an HTTP or HTTPS URL, a link "
    "to it, so the reader "
    "can trace each claim back to its document. List each document once, and use "
    "only titles and links that appeared in the search results — never invent them. "
    "Do not use passage indexes such as [1] or [n] as citations. document_id and "
    "chunk_id are internal tool inputs; never expose them in the answer. If a "
    "document has no accessible URL, list its title only."
)


_KNOWLEDGE_AUX_GUIDANCE = (
    "Use knowledge_list only when discovery or narrowing is useful. When a retrieved "
    "passage is incomplete, ambiguous, or lacks context, call knowledge_read with the "
    "internal document_id or chunk_id. Use knowledge_find for exact identifiers, "
    "error codes, API names, or literal phrases. These IDs are only for tool calls: "
    "never expose them to the user, and never cite passages as [1] or [n]."
)


# Added when the execution sandbox is wired in (code_interpreter and/or shell).
# Reflex-level: reach for execution instead of computing in your head, and split the
# two entrypoints. publish_artifact's own block covers surfacing files, so this does
# not repeat it.
_SANDBOX_GUIDANCE = (
    "You have a remote Agent Loop sandbox that runs real code and shell commands. "
    "This service cannot inspect the user's local filesystem directly; read, grep, "
    "parse, or transform files only inside sandbox mounts such as /mnt/user, "
    "/mnt/skills, and /mnt/system. Reach for the sandbox "
    "whenever execution beats reasoning: non-trivial arithmetic, parsing or "
    "transforming files, data analysis, running or testing a script, checking an "
    "actual command's output — do not compute large or precise results in your head. "
    "Use code_interpreter to run code (e.g. Python) and shell to run shell commands; "
    "they share the same environment and mounts, including the durable per-user "
    "directory at /mnt/user ($AGENT_USER_PATH). For file inspection, prefer shell "
    "commands such as ls, find, rg/grep, sed, head, tail, and cat; for multi-step "
    "parsing or larger transformations, prefer code_interpreter. The shell runs "
    "operating-system and CLI commands only — it is not a way to invoke your own "
    "tools: load_skill, knowledge_search, publish_artifact and the rest are "
    "tool/function calls, so call them directly and never type a tool name as a "
    "shell command. Keep stderr visible when a command fails so you can see why, "
    "and fix and retry rather than guessing at the result."
)


# Added when an Agent explicitly enables code access and has a sandbox tool that
# can explore it. The repository root is fixed at /opt/code in the sandbox image
# (the AGENT_CODE_PATH env var mirrors it for services, but the agent just uses
# the path directly). The knowledge base is the primary ground truth; the code is
# the fallback when it comes up empty on questions about the system's own
# implementation.
_CODE_LAYER_GUIDANCE = (
    "A read-only code layer at `/opt/code` holds "
    "the source repositories behind this system, one per subdirectory. When "
    "knowledge_search / the knowledge base does not answer a question that is "
    "really about how this system's code behaves, fall back to the code: run "
    "`ls /opt/code` to see which repositories are available, then explore the "
    "relevant one with shell / code_interpreter (ripgrep or grep to find "
    "symbols, cat to read files). It is read-only reference material — do not "
    "try to modify it — and it is a fallback for source-level questions, not a "
    "replacement for knowledge_search on document questions."
)


_CAPABILITY_PROMPTS: tuple[CapabilityPrompt, ...] = (
    CapabilityPrompt("web_search", ("web_search",), _WEB_SEARCH_GUIDANCE),
    CapabilityPrompt("knowledge", ("knowledge_search",), _KNOWLEDGE_GUIDANCE),
    CapabilityPrompt("sandbox", ("code_interpreter", "shell"), _SANDBOX_GUIDANCE),
    CapabilityPrompt("file_output", ("publish_artifact",), _FILE_OUTPUT_GUIDANCE),
)


def _code_layer_block(code_manifest: str) -> str:
    """Code repository guidance. With a manifest (the per-agent, admin-curated
    list of what each repo is), lead with it so the model knows the repos up
    front; without one, fall back to discover-by-`ls`."""
    manifest = (code_manifest or "").strip()
    if not manifest:
        return _CODE_LAYER_GUIDANCE
    return (
        "A read-only code layer at `/opt/code` holds the source repositories "
        "behind this system, one per subdirectory. The available repositories:"
        "\n\n" + manifest + "\n\n"
        "When knowledge_search / the knowledge base does not answer a question "
        "that is really about how this system's code behaves, fall back to the "
        "code: open the relevant repository under `/opt/code` and explore it with "
        "shell / code_interpreter (ripgrep or grep to find symbols, cat to read "
        "files); run `ls /opt/code` for anything the list above does not cover. "
        "It is read-only reference material — do not try to modify it — and it "
        "is a fallback for source-level questions, not a replacement for "
        "knowledge_search on document questions."
    )


def _render_capability_guidance(
    *,
    tool_names: List[str],
    aliyun_pai_enabled: bool,
    code_enabled: bool,
    code_manifest: str,
) -> List[str]:
    tool_set = set(tool_names)
    blocks: List[str] = []
    for capability in _CAPABILITY_PROMPTS:
        if not capability.enabled(tool_set):
            continue
        blocks.append(capability.guidance)
        if capability.id == "knowledge" and any(
            name in tool_set for name in KNOWLEDGE_TOOL_NAMES[1:]
        ):
            blocks.append(_KNOWLEDGE_AUX_GUIDANCE)
        if capability.id == "sandbox" and code_enabled:
            blocks.append(_code_layer_block(code_manifest))
    if aliyun_pai_enabled and "shell" in tool_set:
        blocks.append(_ALIYUN_CLI_GUIDANCE)
    return blocks


def render_stable_system_prompt(
    instructions: str, *, tool_names: List[str], project_context: str = "",
    aliyun_pai_enabled: bool = False, code_enabled: bool = False,
    code_manifest: str = "",
) -> str:
    """Stable, cacheable layer: the agent's persona (a single freeform Markdown
    `instructions` document) + project context + the always-on tool protocol/guidance.
    Excludes volatile content (memory, per-request instructions, conversation summary)."""
    parts: List[str] = [instructions.strip() or DEFAULT_INSTRUCTIONS.strip()]
    if project_context.strip():
        parts.append("# Project context\n" + project_context.strip())
    tools_section = "# Tools\n" + _TOOL_PROTOCOL
    if tool_names:
        tools_section += "\n\nTools available this session: " + ", ".join(tool_names) + "."
        for block in _render_capability_guidance(
            tool_names=tool_names,
            aliyun_pai_enabled=aliyun_pai_enabled,
            code_enabled=code_enabled,
            code_manifest=code_manifest,
        ):
            tools_section += "\n\n" + block
    else:
        tools_section += "\n\nYou have no tools enabled in this session; answer from your own knowledge."
    parts.append(tools_section)
    return "\n\n".join(parts)


# Appended (as a stable "# Subagent protocol" block) to a subagent's system prompt.
# A subagent is a delegated worker running in an isolated context (no parent
# conversation), so this reframes its job: do the work with tools, then return ONE
# compact, evidence-bearing summary the coordinator can act on — the whole point of
# the context firewall.
_SUBAGENT_PROTOCOL = (
    "You are a delegated subagent working on ONE self-contained task handed to you "
    "by a coordinating agent. You do NOT see the parent conversation — the task "
    "message is complete on its own, so never ask clarifying questions (there is no "
    "user to answer). Use your tools to actually do the work (search, read, "
    "explore), not to describe or plan it. When finished, reply with a SINGLE final "
    "message that is a compact, self-contained summary: the answer plus the concrete "
    "evidence behind it — knowledge-base source ids, file paths with line ranges, "
    "URLs — enough for the coordinator to act without redoing your work. Be thorough "
    "in the work but terse in the summary; do not dump raw tool output."
)


def render_subagent_system_prompt(
    instructions: str, *, tool_names: List[str], project_context: str = "",
    aliyun_pai_enabled: bool = False, code_enabled: bool = False,
    code_manifest: str = "",
) -> str:
    """A subagent's system prompt: the same stable persona + tool guidance as a
    top-level agent, with the subagent protocol appended so it does the work and
    returns a single compact summary. Reuses ``render_stable_system_prompt`` so
    every capability block (knowledge, code-layer, sandbox, web) stays identical."""
    base = render_stable_system_prompt(
        instructions, tool_names=tool_names, project_context=project_context,
        aliyun_pai_enabled=aliyun_pai_enabled,
        code_enabled=code_enabled, code_manifest=code_manifest,
    )
    return base + "\n\n# Subagent protocol\n" + _SUBAGENT_PROTOCOL


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
