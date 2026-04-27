"""Mini GenericAgent — CLI 入口。
- REPL：读取用户任务 → agent_runner_loop → L4 归档 → 等下一条
- 对话历史跨任务保持（靠 trim_history 防溢出），working memory 逐任务传递
- 启动时把 L1 索引拼到系统提示词末尾，让 Agent 看到可用记忆
"""
import os, sys, re, json, datetime, uuid
from llm_client import LLMClient
from agent_loop import agent_runner_loop
from tools import GenericHandler, SEDIMENT_HOOK
from session_store import SessionStore
from skill_manager import scan_skills, get_skills_prompt, get_use_skill_schema, match_skill, build_skill_user_input

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    import config
except ImportError:
    print('❌ 未找到 config.py。请先 `cp config_template.py config.py` 并填入 API key。')
    sys.exit(1)

TOOLS_SCHEMA = json.load(open(os.path.join(ROOT, 'tools_schema.json'), encoding='utf-8'))
SYS_PROMPT_BASE = open(os.path.join(ROOT, 'prompts', 'sys_prompt.txt'), encoding='utf-8').read()

SKILLS = scan_skills(os.path.join(ROOT, 'skills'))
if SKILLS:
    TOOLS_SCHEMA.append(get_use_skill_schema())
    print(f'[Skills] Loaded {len(SKILLS)} skill(s): {", ".join(SKILLS.keys())}')


def build_system_prompt():
    """系统提示词 = 基础提示 + L1 索引（运行时注入）。"""
    idx_path = os.path.join(ROOT, 'memory', 'global_index.txt')
    idx = open(idx_path, encoding='utf-8').read() if os.path.exists(idx_path) else '(empty)'
    return SYS_PROMPT_BASE + '\n' + idx + get_skills_prompt(SKILLS)


def archive_session(client, task, exit_reason):
    """把对话历史 dump 到 memory/L4_raw_sessions/{时间戳}.md。"""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(ROOT, 'memory', 'L4_raw_sessions', f'{ts}.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# Task ({ts})\n{task}\n\n')
        f.write(f'## Exit\n```json\n{json.dumps(exit_reason, ensure_ascii=False, default=str, indent=2)}\n```\n\n')
        f.write('## History\n```json\n')
        json.dump(client.history, f, ensure_ascii=False, default=str, indent=2)
        f.write('\n```\n')
    print(f'\n📦 会话已归档: {os.path.relpath(path, ROOT)}')


def main():
    api_key = getattr(config, 'API_KEY', None)
    if not api_key or '<your-' in api_key or api_key == '':
        print('❌ config.py 中 API_KEY 未填写或仍为模板值。')
        sys.exit(1)
    api_base = getattr(config, 'API_BASE', None)
    if not api_base or '<your-' in api_base:
        print('❌ config.py 中 API_BASE 未填写或仍为模板值，请填入真实端点。')
        sys.exit(1)

    client = LLMClient(
        api_key=api_key,
        api_base=api_base,
        model=getattr(config, 'MODEL', 'qwen-plus'),
        max_tokens=getattr(config, 'MAX_TOKENS', 8192),
        history_trim_tokens=getattr(config, 'HISTORY_TRIM_TOKENS', 80000),
        timeout=getattr(config, 'TIMEOUT', 300),
    )
    max_turns = getattr(config, 'MAX_TURNS', 40)
    store = SessionStore(os.path.join(ROOT, 'memory', 'sessions'))
    session_id = str(uuid.uuid4())
    prev_handler = None

    def save_session(handler=None):
        handler = handler or prev_handler
        handler_state = None
        if handler:
            handler_state = {
                'history_info': list(handler.history_info),
                'working': dict(handler.working),
            }
        store.save(
            session_id=session_id,
            llm_history=list(client.history),
            ui_messages=[],
            handler_state=handler_state,
        )

    client.history_changed = save_session
    save_session()

    print(f'\n\033[36m{"="*60}\033[0m')
    print(f'\033[1mMini GenericAgent\033[0m  model={client.model}  max_turns={max_turns}')
    print(f'端点: {api_base}')
    print(f"工作目录: {os.getcwd()}   记忆库: {os.path.relpath(os.path.join(ROOT, 'memory'))}")
    print(f'\033[36m{"="*60}\033[0m')
    print('输入任务回车启动；空行 / exit / quit / Ctrl-D 退出。\n')

    while True:
        try:
            task = input('\033[36m>>> \033[0m').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task or task in ('exit', 'quit'):
            break

        # Slash command → skill dispatch
        sk, sk_args = match_skill(task, SKILLS)
        if sk:
            task = build_skill_user_input(sk, sk_args)
            print(f'\033[35m[Skill] {sk.name} activated\033[0m')

        handler = GenericHandler(cwd=os.getcwd(), mini_agent_root=ROOT)
        if sk:
            handler.working['active_skill'] = sk.name
            handler.working['related_sop'] = f'skills/{sk.name}/SKILL.md'
        if prev_handler:
            handler.history_info = list(prev_handler.history_info)
            if 'key_info' in prev_handler.working:
                ki = re.sub(r'\n\[SYSTEM\] 此为.*?工作记忆[。\n]*', '', prev_handler.working['key_info'])
                handler.working['key_info'] = ki
                ps = prev_handler.working.get('passed_sessions', 0) + 1
                handler.working['passed_sessions'] = ps
                handler.working['key_info'] += f'\n[SYSTEM] 此为 {ps} 个对话前设置的key_info，若已在新任务，先更新或清除工作记忆。\n'
        handler.history_info.append(f"[USER]: {task[:200]}")

        user_input = task
        if prev_handler:
            user_input = handler._anchor_prompt() + f"\n\n### 用户当前消息\n{task}"

        # 注入沉淀评估 hook：强制任务收尾时走一次记忆判断。
        handler._done_hooks.append(SEDIMENT_HOOK)

        try:
            exit_reason = agent_runner_loop(
                client=client,
                system_prompt=build_system_prompt(),
                user_input=user_input,
                handler=handler,
                tools_schema=TOOLS_SCHEMA,
                max_turns=max_turns,
            )
        except KeyboardInterrupt:
            print('\n[!] 用户中断本任务')
            exit_reason = {'result': 'INTERRUPTED'}
        except Exception as e:
            import traceback
            traceback.print_exc()
            exit_reason = {'result': 'ERROR', 'msg': str(e)}

        try:
            archive_session(client, task, exit_reason)
        except Exception as e:
            print(f'[Warn] 归档失败: {e}')
        try:
            save_session(handler)
        except Exception as e:
            print(f'[Warn] 会话持久化失败: {e}')
        prev_handler = handler
        print()


if __name__ == '__main__':
    main()
