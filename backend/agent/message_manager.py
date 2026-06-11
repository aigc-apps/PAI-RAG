from typing import List
from dataclasses import dataclass, field
from loguru import logger
from memory.utils import estimate_tokens_in_text, truncate, get_tokenizer
from common.llm.models import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS


TOOL_RESULT_TRUNCATED_MARKER = "\n...[content truncated]"
TOOL_GROUP_SUMMARY_TEMPLATE = "Called {tool_name}({args_preview}), result: {result_preview}"
DEFAULT_MAX_TOOL_RESULT_TOKENS = 5000
DEFAULT_RESERVE_RATIO = 0.1
DEFAULT_TRUNCATED_TOOL_RESULT_TOKENS = 200
DEFAULT_MIN_PROTECTED_HISTORY_ROUNDS = 5
DEFAULT_HISTORY_MSG_MAX_TOKENS = 1500
MESSAGE_OVERHEAD_TOKENS = 4


@dataclass
class MessageGroup:
    messages: List[dict] = field(default_factory=list)
    tokens: int = 0
    group_type: str = ""  # "system" | "user" | "assistant_text" | "tool_round"
    protected: bool = False


class AgentMessageManager:
    def __init__(
        self,
        context_window: int,
        max_output_tokens: int,
        reserve_ratio: float = DEFAULT_RESERVE_RATIO,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        min_protected_history_rounds: int = DEFAULT_MIN_PROTECTED_HISTORY_ROUNDS,
        history_msg_max_tokens: int = DEFAULT_HISTORY_MSG_MAX_TOKENS,
    ):
        # 为了backward compatibility，如果context_window小于等于10000，则使用默认值
        if not context_window or context_window <= 10000:
            context_window = DEFAULT_CONTEXT_WINDOW
        # 为了backward compatibility，如果max_output_tokens小于等于0，则使用默认值
        if not max_output_tokens or max_output_tokens <= 0:
            max_output_tokens = DEFAULT_MAX_TOKENS
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.max_tool_result_tokens = max_tool_result_tokens
        self.min_protected_history_rounds = min_protected_history_rounds
        self.history_msg_max_tokens = history_msg_max_tokens
        self.token_budget = int(
            context_window - max_output_tokens - context_window * reserve_ratio
        )
        self.tokenizer = get_tokenizer()
        logger.info(
            f"AgentMessageManager initialized: context_window={context_window}, "
            f"max_output={max_output_tokens}, token_budget={self.token_budget}"
        )

    def estimate_msg_tokens(self, msg: dict) -> int:
        tokens = MESSAGE_OVERHEAD_TOKENS
        content = msg.get("content") or ""
        if isinstance(content, str):
            tokens += estimate_tokens_in_text(content, tokenizer=self.tokenizer)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    tokens += estimate_tokens_in_text(
                        item.get("text", ""), tokenizer=self.tokenizer
                    )
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tokens += estimate_tokens_in_text(
                    str(tc), tokenizer=self.tokenizer
                )
        return tokens

    def estimate_messages_tokens(self, messages: List[dict]) -> int:
        return sum(self.estimate_msg_tokens(m) for m in messages)

    def cap_tool_result(self, content: str) -> str:
        if not content:
            return content
        tokens = estimate_tokens_in_text(content, tokenizer=self.tokenizer)
        if tokens <= self.max_tool_result_tokens:
            return content
        truncated_text, _ = truncate(
            content,
            max_token=self.max_tool_result_tokens,
            tokenizer=self.tokenizer,
        )
        logger.info(
            f"Tool result truncated from {tokens} to ~{self.max_tool_result_tokens} tokens"
        )
        return truncated_text + TOOL_RESULT_TRUNCATED_MARKER

    def group_messages(self, messages: List[dict]) -> List[MessageGroup]:
        groups = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "system":
                g = MessageGroup(
                    messages=[msg],
                    tokens=self.estimate_msg_tokens(msg),
                    group_type="system",
                    protected=True,
                )
                groups.append(g)
                i += 1

            elif role == "user":
                g = MessageGroup(
                    messages=[msg],
                    tokens=self.estimate_msg_tokens(msg),
                    group_type="user",
                    protected=False,
                )
                groups.append(g)
                i += 1

            elif role == "assistant":
                has_tool_calls = bool(msg.get("tool_calls"))
                if has_tool_calls:
                    group_msgs = [msg]
                    group_tokens = self.estimate_msg_tokens(msg)
                    j = i + 1
                    while j < len(messages) and messages[j].get("role") == "tool":
                        group_msgs.append(messages[j])
                        group_tokens += self.estimate_msg_tokens(messages[j])
                        j += 1
                    g = MessageGroup(
                        messages=group_msgs,
                        tokens=group_tokens,
                        group_type="tool_round",
                        protected=False,
                    )
                    groups.append(g)
                    i = j
                else:
                    g = MessageGroup(
                        messages=[msg],
                        tokens=self.estimate_msg_tokens(msg),
                        group_type="assistant_text",
                        protected=False,
                    )
                    groups.append(g)
                    i += 1

            elif role == "tool":
                g = MessageGroup(
                    messages=[msg],
                    tokens=self.estimate_msg_tokens(msg),
                    group_type="tool_round",
                    protected=False,
                )
                groups.append(g)
                i += 1

            else:
                g = MessageGroup(
                    messages=[msg],
                    tokens=self.estimate_msg_tokens(msg),
                    group_type="unknown",
                    protected=False,
                )
                groups.append(g)
                i += 1

        self._mark_protected(groups)
        return groups

    def _mark_protected(self, groups: List[MessageGroup]):
        for g in groups:
            if g.group_type == "system":
                g.protected = True

        rounds_seen = 0
        for g in reversed(groups):
            if g.group_type == "user":
                rounds_seen += 1
            if rounds_seen <= self.min_protected_history_rounds:
                g.protected = True
            else:
                break

        for g in reversed(groups):
            if g.group_type in ("tool_round", "assistant_text"):
                g.protected = True
                break

    def _truncate_tool_results_in_group(
        self,
        group: MessageGroup,
        target_tokens: int = DEFAULT_TRUNCATED_TOOL_RESULT_TOKENS,
    ) -> int:
        saved = 0
        for msg in group.messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            if not content:
                continue
            current_tokens = estimate_tokens_in_text(content, tokenizer=self.tokenizer)
            if current_tokens <= target_tokens:
                continue
            truncated_text, new_tokens = truncate(
                content, max_token=target_tokens, tokenizer=self.tokenizer
            )
            msg["content"] = truncated_text + TOOL_RESULT_TRUNCATED_MARKER
            saved += current_tokens - new_tokens
        group.tokens -= saved
        return saved

    def _summarize_tool_group(self, group: MessageGroup) -> int:
        if group.group_type != "tool_round":
            return 0
        if len(group.messages) < 2:
            return 0

        old_tokens = group.tokens
        assistant_msg = group.messages[0]
        tool_calls = assistant_msg.get("tool_calls", [])

        summaries = []
        tool_msgs = [m for m in group.messages if m.get("role") == "tool"]

        for tc in tool_calls:
            tool_name = ""
            args_preview = ""
            if hasattr(tc, "function"):
                tool_name = tc.function.name or "unknown"
                args_str = tc.function.arguments or ""
            elif isinstance(tc, dict):
                fn = tc.get("function", {})
                tool_name = fn.get("name", "unknown")
                args_str = fn.get("arguments", "")
            else:
                tool_name = "unknown"
                args_str = ""

            if len(args_str) > 100:
                args_preview = args_str[:100] + "..."
            else:
                args_preview = args_str

            tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
            result_preview = ""
            for tm in tool_msgs:
                if tm.get("tool_call_id") == tc_id:
                    result_text = tm.get("content") or ""
                    if len(result_text) > 200:
                        result_preview = result_text[:200] + "..."
                    else:
                        result_preview = result_text
                    break

            summaries.append(
                TOOL_GROUP_SUMMARY_TEMPLATE.format(
                    tool_name=tool_name,
                    args_preview=args_preview,
                    result_preview=result_preview,
                )
            )

        summary_content = "\n".join(summaries)
        summary_msg = {"role": "assistant", "content": summary_content}
        group.messages = [summary_msg]
        group.tokens = self.estimate_msg_tokens(summary_msg)
        group.group_type = "assistant_text"

        saved = old_tokens - group.tokens
        return saved

    def fit_to_budget(self, messages: List[dict]) -> List[dict]:
        total_tokens = self.estimate_messages_tokens(messages)
        logger.info(
            f"Context check: {total_tokens}/{self.token_budget} tokens "
            f"({len(messages)} messages), "
            f"{'within budget' if total_tokens <= self.token_budget else 'over budget, compressing'}"
        )
        if total_tokens <= self.token_budget:
            return messages

        groups = self.group_messages(messages)
        total_tokens = sum(g.tokens for g in groups)

        if total_tokens <= self.token_budget:
            return self._flatten_groups(groups)

        compressible = [
            g for g in groups
            if not g.protected and g.group_type == "tool_round"
        ]
        for g in compressible:
            if total_tokens <= self.token_budget:
                break
            saved = self._truncate_tool_results_in_group(g)
            total_tokens -= saved
            if saved > 0:
                logger.info(f"L1 truncation saved {saved} tokens")

        if total_tokens <= self.token_budget:
            return self._flatten_groups(groups)

        for g in compressible:
            if total_tokens <= self.token_budget:
                break
            saved = self._summarize_tool_group(g)
            total_tokens -= saved
            if saved > 0:
                logger.info(f"L2 summarization saved {saved} tokens")

        if total_tokens <= self.token_budget:
            return self._flatten_groups(groups)

        non_protected = [i for i, g in enumerate(groups) if not g.protected]
        for idx in non_protected:
            if total_tokens <= self.token_budget:
                break
            g = groups[idx]
            total_tokens -= g.tokens
            g.messages = []
            g.tokens = 0
            logger.info(f"L3 dropped group type={g.group_type}")

        if total_tokens > self.token_budget:
            saved = self._truncate_protected_history(groups)
            total_tokens -= saved

        if total_tokens > self.token_budget:
            logger.warning(
                f"After all compression, still over budget: "
                f"{total_tokens}/{self.token_budget}"
            )

        return self._flatten_groups(groups)

    def _truncate_msg_content(self, msg: dict, max_tokens: int) -> int:
        content = msg.get("content") or ""
        if not isinstance(content, str) or not content:
            return 0
        current_tokens = estimate_tokens_in_text(content, tokenizer=self.tokenizer)
        if current_tokens <= max_tokens:
            return 0
        truncated_text, new_tokens = truncate(
            content, max_token=max_tokens, tokenizer=self.tokenizer
        )
        msg["content"] = truncated_text + TOOL_RESULT_TRUNCATED_MARKER
        return current_tokens - new_tokens

    def _truncate_protected_history(self, groups: List[MessageGroup]) -> int:
        total_saved = 0
        for g in groups:
            if g.group_type == "system":
                continue
            if not g.protected:
                continue
            for msg in g.messages:
                role = msg.get("role", "")
                if role in ("user", "assistant", "tool"):
                    saved = self._truncate_msg_content(
                        msg, self.history_msg_max_tokens
                    )
                    if saved > 0:
                        g.tokens -= saved
                        total_saved += saved
        if total_saved > 0:
            logger.info(f"L4 truncated protected history, saved {total_saved} tokens")
        return total_saved

    def _flatten_groups(self, groups: List[MessageGroup]) -> List[dict]:
        result = []
        for g in groups:
            result.extend(g.messages)
        return result
