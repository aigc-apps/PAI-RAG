# Tool Call Title Summaries Design

## Goal

Make collapsed tool-call records distinguishable at a glance. Every built-in tool has a localized, human-readable name, retains its raw tool name for technical clarity, and shows a concise summary of its most important arguments when available.

The title format is:

`Localized name (raw_tool_name) · argument summary`

Examples:

- Chinese: `执行命令 (shell) · ls /opt/code/easyrec/`
- English: `Run command (shell) · ls /opt/code/easyrec/`

## Scope

This is a frontend-only presentation change. It does not alter tool schemas, stream events, persisted conversations, backend serialization, or the expanded argument and result sections.

The feature covers every built-in tool currently defined under `backend/agent/tools/builtin`. Unknown or custom tools remain supported through an explicit fallback.

## Display Configuration

The frontend maintains one centralized configuration keyed by raw tool name. Each entry provides:

- an i18n key for the localized display name;
- a formatter that parses the tool's arguments and returns an optional summary.

The formatter mappings are:

| Tool | Localized meaning | Summary fields |
| --- | --- | --- |
| `shell` | Run command | `command` |
| `code_interpreter` | Run code | `language`, then the first line of `code` |
| `web_search` | Search the web | `query` |
| `web_fetch` | Fetch webpage | `url` |
| `knowledge_search` | Search knowledge | `query` |
| `knowledge_find` | Find in knowledge | `query` |
| `knowledge_read` | Read knowledge document | `document_id`, falling back to `chunk_id` |
| `knowledge_list` | List knowledge bases | no summary |
| `current_datetime` | Get current time | no summary |
| `load_skill` | Load skill | `skill_id` |
| `enable_skill_for_agent` | Configure agent skill | `skill_id` |
| `read_skill_resource` | Read skill resource | `skill_id`, then `path` |
| `install_skill` | Install skill | source `type`, then `url`, Git `path`, or `upload_id` when present |
| `publish_artifact` | Publish artifact | `name`, falling back to `path` |
| `spawn_subagent` | Start subagent | `agent_id`, then `task` |
| `read_handle` | Read stored result | `handle` |

Multiple summary values are joined with ` · `. Formatters only display the fields listed above; they do not dump arbitrary objects or arrays into the title.

## Rendering and Visual Hierarchy

Within the existing tool-call trigger:

1. The localized display name is the primary label.
2. The raw tool name appears in parentheses with weaker visual emphasis.
3. The argument summary appears after `·` with the weakest emphasis.
4. Status and duration follow the compact alignment rules below.

The trigger stays on one line. The name, status dot, duration, and chevron do not shrink. The summary occupies the remaining width, truncates with an ellipsis, and exposes its full normalized text through the native hover title.

### Status and Duration

Each row uses the leading status dot as its only visible compact status indicator:

- a pulsing accent-colored dot means running;
- a green dot means completed;
- a red dot means failed.

The visible `Running`, `Done`, and `Failed` text beside each tool title is removed because it duplicates the dot and competes with the argument summary. The dot exposes the localized status through an accessible label so state is not conveyed by color alone.

Completed and failed durations occupy a fixed-width, right-aligned, tabular-number column immediately before the chevron. This keeps durations vertically aligned across all tool rows while allowing the argument summary to consume and truncate within the flexible middle region. Running tools have an empty duration column until elapsed time is available.

Failed tools retain the existing red border and tinted background, auto-expand on first render, and show the localized execution-failure card with the concrete error. If the user manually collapses the row, the red dot and error container styling continue to identify the failure.

Argument text is normalized before display by replacing line breaks and runs of whitespace with a single space and trimming the result. The expanded details continue to show the original argument JSON unchanged.

## Fallback and Error Behavior

- Invalid JSON, non-object JSON, missing fields, empty strings, or a formatter failure produce no summary and never break the tool card.
- A known tool without a usable summary still displays `Localized name (raw_tool_name)`.
- An unknown or custom tool preserves the current behavior and displays only its raw tool name.
- Tools with no meaningful arguments intentionally omit the separator and summary.
- A formatter must be pure and must not mutate the tool-call record.

## Internationalization

Each built-in tool receives matching keys in the existing English and Chinese dictionaries. The component resolves the localized name with the current application language, so changing language immediately updates existing tool-call titles.

Raw tool names and argument values are never translated.

## Testing

Frontend component tests cover:

- the complete title for a built-in tool in Chinese;
- the corresponding English title after switching language;
- primary-field fallback such as `knowledge_read` choosing `chunk_id` when `document_id` is absent;
- multi-field summaries such as `code_interpreter` and `read_skill_resource`;
- whitespace normalization and the full hover title for long or multiline values;
- known tools with no summary;
- invalid, non-object, missing, and empty arguments;
- unknown-tool fallback that preserves the raw name only.
- status-dot accessible labels in both supported languages;
- removal of redundant visible status text;
- a fixed-width, right-aligned duration column before the chevron;
- retained error auto-expansion and visible failure details.

The frontend build and complete frontend test suite must pass after implementation.

## Out of Scope

- Backend-generated display metadata.
- Translating argument values or raw tool names.
- Displaying secondary tuning arguments such as timeouts, limits, retrieval modes, or offsets.
- Changing expanded tool details, status semantics, error auto-expansion behavior, duration calculation, or stream persistence.
- Adding display configuration for third-party or dynamically registered custom tools.
