"""7 个原子工具 + Handler。
移植自 ga.py，去除 PowerShell / 浏览器 / Windows 特化代码。
每个 do_xxx 返回 StepOutcome；中间打印直接 print。
"""
import os, re, sys, time, json, threading, subprocess, tempfile, itertools, collections, difflib
from pathlib import Path
from agent_loop import BaseHandler, StepOutcome


# 任务"形式上完成"后强制执行的沉淀评估提示。
# 由前端入口按需 push 到 handler._done_hooks，
# agent_loop 在 CURRENT_TASK_DONE 出口拦截弹出，让 LLM 必走一次"该不该沉淀"判断。
SEDIMENT_HOOK = (
    '[INTERNAL] 任务已收尾。请评估是否值得长期记忆：\n'
    '- 若本次任务过程中发现了新的环境事实/路径/凭证，或摸索出非平凡的步骤序列（被坑过的经验）'
    '→ 调用 `start_long_term_update`\n'
    '- 若纯属常规问答 / 信息已记录 / 任务过短（< 5 轮）'
    '→ 直接 `<summary>无需沉淀</summary>` 结束\n'
    '判断标准从严，避免污染长期记忆。'
)


# ──────────────────────────── 通用工具函数 ──────────────────────────── #

def smart_format(data, max_str_len=100, omit_str=' ... '):
    """长字符串保留头尾，省略中部。"""
    if not isinstance(data, str):
        data = str(data)
    if len(data) < max_str_len + len(omit_str) * 2:
        return data
    return f"{data[:max_str_len // 2]}{omit_str}{data[-max_str_len // 2:]}"


def expand_file_refs(text, base_dir=None):
    """展开 {{file:路径:起始行:结束行}} 引用为实际内容。"""
    pattern = r'\{\{file:(.+?):(\d+):(\d+)\}\}'

    def replacer(m):
        path, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        path = os.path.abspath(os.path.join(base_dir or '.', path))
        if not os.path.isfile(path):
            raise ValueError(f"引用文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if start < 1 or end > len(lines) or start > end:
            raise ValueError(f"行号越界: {path} 共{len(lines)}行, 请求{start}-{end}")
        return ''.join(lines[start - 1:end])

    return re.sub(pattern, replacer, text)


# ──────────────────────────── 原子工具实现 ──────────────────────────── #

def code_run(code, code_type='python', timeout=60, cwd=None, cancel_evt=None):
    """同步执行 python 或 bash，流式打印 stdout。cancel_evt 触发即 kill 子进程。"""
    cwd = cwd or os.getcwd()
    os.makedirs(cwd, exist_ok=True)
    tmp_path = None
    if code_type in ('python', 'py'):
        tmp = tempfile.NamedTemporaryFile(suffix='.ai.py', delete=False, mode='w', encoding='utf-8')
        tmp.write(code)
        tmp_path = tmp.name
        tmp.close()
        cmd = [sys.executable, '-X', 'utf8', '-u', tmp_path]
    elif code_type in ('bash', 'sh', 'shell'):
        cmd = ['bash', '-c', code]
    else:
        return {'status': 'error', 'msg': f'不支持的类型: {code_type}'}

    full = []

    def reader(proc):
        for line_bytes in iter(proc.stdout.readline, b''):
            try:
                line = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                line = line_bytes.decode('utf-8', errors='replace')
            full.append(line)
            try:
                print(line, end='')
            except Exception:
                pass

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=0, cwd=cwd)
        t = threading.Thread(target=reader, args=(proc,), daemon=True)
        t.start()
        start = time.time()
        while t.is_alive():
            if cancel_evt is not None and cancel_evt.is_set():
                proc.kill()
                full.append('\n[Cancelled] 用户中止')
                break
            if time.time() - start > timeout:
                proc.kill()
                full.append('\n[Timeout Error] 超时强制终止')
                break
            time.sleep(0.1)
        t.join(timeout=1)
        exit_code = proc.poll()
        stdout = ''.join(full)
        return {
            'status': 'success' if exit_code == 0 else 'error',
            'exit_code': exit_code,
            'stdout': smart_format(stdout, max_str_len=10000,
                                   omit_str='\n\n[omitted long output]\n\n'),
        }
    except Exception as e:
        return {'status': 'error', 'msg': str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def file_patch(path, old_content, new_content):
    """精确唯一性匹配替换。"""
    path = str(Path(path).resolve())
    if not os.path.exists(path):
        return {'status': 'error', 'msg': '文件不存在'}
    if not old_content:
        return {'status': 'error', 'msg': 'old_content 为空'}
    text = open(path, 'r', encoding='utf-8').read()
    cnt = text.count(old_content)
    if cnt == 0:
        return {'status': 'error',
                'msg': '未找到匹配。请先 file_read 确认当前内容，再分小段 patch。'}
    if cnt > 1:
        return {'status': 'error',
                'msg': f'找到 {cnt} 处匹配，不唯一。请提供更长 old_content 或包含上下文行。'}
    open(path, 'w', encoding='utf-8').write(text.replace(old_content, new_content))
    return {'status': 'success', 'msg': '文件局部修改成功'}


_read_dirs = set()


def _scan_files(base, depth=2):
    try:
        for e in os.scandir(base):
            if e.is_file():
                yield (e.name, e.path)
            elif depth > 0 and e.is_dir(follow_symlinks=False):
                yield from _scan_files(e.path, depth - 1)
    except (PermissionError, OSError):
        pass


def file_read(path, start=1, keyword=None, count=200, show_linenos=True):
    """流式按行读取，支持 keyword 上下文搜索 + 长行截断 + 模糊路径建议。"""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            stream = ((i, l.rstrip('\r\n')) for i, l in enumerate(f, 1))
            stream = itertools.dropwhile(lambda x: x[0] < start, stream)
            if keyword:
                before = collections.deque(maxlen=count // 3)
                for i, l in stream:
                    if keyword.lower() in l.lower():
                        res = (list(before) + [(i, l)]
                               + list(itertools.islice(stream, count - len(before) - 1)))
                        break
                    before.append((i, l))
                else:
                    return (f"Keyword '{keyword}' not found after line {start}. "
                            f"Falling back to content from line {start}:\n\n"
                            + file_read(path, start, None, count, show_linenos))
            else:
                res = list(itertools.islice(stream, count))
            realcnt = len(res)
            L_MAX = min(max(100, 256000 // max(realcnt, 1)), 8000)
            TAG = ' ... [TRUNCATED]'
            remaining = sum(1 for _ in itertools.islice(stream, 5000))
            total = (res[0][0] - 1 if res else start - 1) + realcnt + remaining
            total_tag = f"[FILE] Total {total}{'+' if remaining >= 5000 else ''} lines\n"
            res = [(i, l if len(l) <= L_MAX else l[:L_MAX] + TAG) for i, l in res]
            out = '\n'.join(f"{i}|{l}" if show_linenos else l for i, l in res)
            if show_linenos:
                out = total_tag + out
            _read_dirs.add(os.path.dirname(os.path.abspath(path)))
            return out
    except FileNotFoundError:
        msg = f'Error: File not found: {path}'
        try:
            tgt = os.path.basename(path)
            scan = os.path.dirname(os.path.dirname(os.path.abspath(path)))
            roots = [scan] + [d for d in _read_dirs if not d.startswith(scan)]
            cands = list(itertools.islice(
                (c for base in roots for c in _scan_files(base)), 2000))
            top = sorted([(difflib.SequenceMatcher(None, tgt.lower(), c[0].lower()).ratio(), c)
                          for c in cands], key=lambda x: -x[0])[:5]
            top = [(s, c) for s, c in top if s > 0.3]
            if top:
                msg += '\n\nDid you mean:\n' + '\n'.join(f'  {c[1]}  ({s:.0%})' for s, c in top)
        except Exception:
            pass
        return msg
    except Exception as e:
        return f'Error: {e}'


# ──────────────────────────── Handler ──────────────────────────── #

class GenericHandler(BaseHandler):
    """工具分发 + 工作记忆 + 历史摘要。"""

    def __init__(self, cwd, mini_agent_root):
        self.cwd = os.path.abspath(cwd)
        self.root = mini_agent_root          # 用于定位 prompts/memory 目录
        self.working = {}                    # key_info / related_sop / passed_sessions
        self.history_info = []               # 每轮的 <summary> 摘要
        self.current_turn = 0
        self.max_turns = 40
        self.cancel_evt = None               # 前端可注入 threading.Event 用于中止
        self._done_hooks = []                # 任务完成前必须执行的 prompt 队列

    # ── 路径与代码块抽取 ──
    def _abs(self, path):
        return os.path.abspath(os.path.join(self.cwd, path)) if path else ''

    @staticmethod
    def _extract_code_block(text, code_type):
        kind = {'python': 'python|py', 'bash': 'bash|sh|shell'}.get(code_type, re.escape(code_type))
        m = re.findall(rf'```(?:{kind})\n(.*?)\n```', text or '', re.DOTALL)
        return m[-1].strip() if m else None

    # ── working memory ──
    def _anchor_prompt(self, skip=False):
        if skip:
            return '\n'
        h_str = '\n'.join(self.history_info[-20:])
        out = f'\n### [WORKING MEMORY]\n<history>\n{h_str}\n</history>'
        out += f'\nCurrent turn: {self.current_turn}\n'
        if self.working.get('key_info'):
            out += f"\n<key_info>{self.working['key_info']}</key_info>"
        if self.working.get('related_sop'):
            out += f"\n有不清晰的地方请再次读取 {self.working['related_sop']}"
        return out

    # ── 7 个工具 ──
    def do_code_run(self, args, response):
        code_type = args.get('type', 'python')
        code = args.get('script') or self._extract_code_block(response.content, code_type)
        if not code:
            return StepOutcome('[Error] code missing — provide `script` or a ```python/bash block.',
                               next_prompt='\n')
        timeout = args.get('timeout', 60)
        cwd = os.path.abspath(os.path.join(self.cwd, args.get('cwd', '.')))
        preview = (code[:60].replace('\n', ' ') + ('...' if len(code) > 60 else ''))
        print(f"[Action] Running {code_type} in {os.path.basename(cwd) or cwd}: {preview}")
        result = code_run(code, code_type, timeout, cwd, cancel_evt=self.cancel_evt)
        icon = {'success': '✅', 'error': '❌'}.get(result.get('status'), '⏳')
        snippet = smart_format(result.get('stdout', ''), max_str_len=600,
                               omit_str='\n\n[omitted long output]\n\n')
        print(f"[Status] {icon} exit={result.get('exit_code')}\n[Stdout]\n{snippet}")
        return StepOutcome(result, next_prompt=self._anchor_prompt(skip=args.get('_index', 0) > 0))

    def do_file_read(self, args, response):
        path = self._abs(args.get('path', ''))
        print(f'[Action] Reading file: {path}')
        result = file_read(
            path, start=args.get('start', 1), keyword=args.get('keyword'),
            count=args.get('count', 200), show_linenos=args.get('show_linenos', True),
        )
        if args.get('show_linenos', True) and not result.startswith('Error:'):
            result = '由于设置了 show_linenos，以下返回信息为：(行号|)内容\n' + result
        if ' ... [TRUNCATED]' in result:
            result += '\n\n（某些行被截断，如需完整内容可改用 code_run 读取）'
        result = smart_format(result, max_str_len=20000,
                              omit_str='\n\n[omitted long content]\n\n')
        next_prompt = self._anchor_prompt(skip=args.get('_index', 0) > 0)
        if 'memory' in path or 'sop' in path.lower():
            next_prompt += ('\n[SYSTEM TIPS] 正在读取记忆/SOP 文件。若决定按 SOP 执行，'
                            '请提取关键点 update_working_checkpoint。')
        return StepOutcome(result, next_prompt=next_prompt)

    def do_file_patch(self, args, response):
        path = self._abs(args.get('path', ''))
        old = args.get('old_content', '')
        new = args.get('new_content', '')
        try:
            new = expand_file_refs(new, base_dir=self.cwd)
        except ValueError as e:
            print(f'[Status] ❌ 引用展开失败: {e}')
            return StepOutcome({'status': 'error', 'msg': str(e)}, next_prompt='\n')
        print(f'[Action] Patching: {path}')
        result = file_patch(path, old, new)
        print(f"[Status] {'✅' if result['status'] == 'success' else '❌'} {result['msg']}")
        return StepOutcome(result, next_prompt=self._anchor_prompt(skip=args.get('_index', 0) > 0))

    def do_file_write(self, args, response):
        path = self._abs(args.get('path', ''))
        mode = args.get('mode', 'overwrite')
        action = {'prepend': 'Prepending to', 'append': 'Appending to'}.get(mode, 'Overwriting')
        print(f'[Action] {action}: {path}')

        # 优先 <file_content> 标签，其次首尾 ``` 围栏内容
        text = response.content or ''
        m = re.search(r'<file_content[^>]*>(.*)</file_content>', text, re.DOTALL)
        if m:
            blocks = m.group(1).strip()
        else:
            s, e = text.find('```'), text.rfind('```')
            blocks = text[text.find('\n', s) + 1:e].strip() if -1 < s < e else None

        if not blocks:
            print('[Status] ❌ 未找到 <file_content> 或代码块内容')
            return StepOutcome(
                {'status': 'error', 'msg': '请把内容放进 <file_content>...</file_content> 或代码块'},
                next_prompt='\n')
        try:
            content = expand_file_refs(blocks, base_dir=self.cwd)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            if mode == 'prepend':
                old = open(path, 'r', encoding='utf-8').read() if os.path.exists(path) else ''
                open(path, 'w', encoding='utf-8').write(content + old)
            else:
                with open(path, 'a' if mode == 'append' else 'w', encoding='utf-8') as f:
                    f.write(content)
            print(f'[Status] ✅ {mode} 成功 ({len(content)} bytes)')
            return StepOutcome(
                {'status': 'success', 'writed_bytes': len(content)},
                next_prompt=self._anchor_prompt(skip=args.get('_index', 0) > 0))
        except Exception as e:
            print(f'[Status] ❌ 写入异常: {e}')
            return StepOutcome({'status': 'error', 'msg': str(e)}, next_prompt='\n')

    def do_ask_user(self, args, response):
        question = args.get('question', '请提供输入：')
        candidates = args.get('candidates') or []
        print(f'\n\033[33m[Agent 提问]\033[0m {question}')
        if candidates:
            for i, c in enumerate(candidates, 1):
                print(f'  {i}. {c}')
        try:
            answer = input('\033[33m你的回答 > \033[0m').strip()
        except (EOFError, KeyboardInterrupt):
            answer = '[用户中止]'
        # 数字快捷选项
        if answer.isdigit() and candidates and 1 <= int(answer) <= len(candidates):
            answer = candidates[int(answer) - 1]
        return StepOutcome({'status': 'answered', 'answer': answer},
                           next_prompt=f'用户回答：{answer}\n根据答案继续推进任务。')

    def do_update_working_checkpoint(self, args, response):
        if 'key_info' in args:
            self.working['key_info'] = args['key_info']
        if 'related_sop' in args:
            self.working['related_sop'] = args['related_sop']
        self.working['passed_sessions'] = 0
        print(f"[Info] working memory updated: key_info={smart_format(args.get('key_info', ''), 80)}")
        return StepOutcome({'result': 'working memory updated'},
                           next_prompt=self._anchor_prompt(skip=args.get('_index', 0) > 0))

    def do_start_long_term_update(self, args, response):
        sop_path = os.path.join(self.root, 'prompts', 'memory_management_sop.md')
        sop_text = open(sop_path, 'r', encoding='utf-8').read() if os.path.exists(sop_path) \
            else '(SOP file missing — skip memory update)'
        index_path = os.path.join(self.root, 'memory', 'global_index.txt')
        facts_path = os.path.join(self.root, 'memory', 'global_facts.txt')
        skill_hint = ''
        if self.working.get('active_skill'):
            skill_name = self.working['active_skill']
            skill_hint = (
                f'\n**注意**：本次任务在 Skill `{skill_name}` 下执行。'
                f'沉淀经验时：\n'
                f'- SOP 文件建议命名为 `memory/{skill_name}_sop.md`\n'
                f'- L1 索引中关联标注：`{skill_name} → skills/{skill_name} + memory/{skill_name}_sop.md`\n'
                f'- 只记录 Skill 指令中未覆盖的踩坑经验，不要复制 SKILL.md 的内容\n'
            )
        prompt = (
            '### [总结提炼经验]\n'
            '请按下方 SOP 提取本次任务中【行动验证成功且长期有效】的信息更新长期记忆。\n'
            '**禁止**：临时变量、推理过程、未验证信息、通用常识。\n'
            f'{skill_hint}'
            '**操作步骤**：\n'
            f'1. file_read {index_path} 看现有索引\n'
            f'2. file_read {facts_path} 看现有事实\n'
            '3. 按 SOP 决策树分类信息\n'
            '4. file_patch 最小化更新（绝不 overwrite）\n'
            '5. 若新增 L3 SOP，file_write `memory/<场景>_sop.md` 并 file_patch L1 加导航行\n'
            '6. 无新内容直接结束\n\n'
            '## 记忆更新 SOP（L0）\n' + sop_text
        )
        print('[Info] 开始长期记忆结算流程')
        return StepOutcome({'status': 'sop_loaded'}, next_prompt=prompt)

    def do_use_skill(self, args, response):
        """LLM 主动调用技能。"""
        from skill_manager import scan_skills, build_skill_user_input
        skills = scan_skills(os.path.join(self.root, 'skills'))
        name = args.get('skill', '')
        if name not in skills:
            avail = ', '.join(skills.keys()) if skills else '(none)'
            return StepOutcome(
                {'status': 'error', 'msg': f'Unknown skill: {name}. Available: {avail}'},
                next_prompt='\n')
        sk = skills[name]
        skill_args = args.get('args', '')
        self.working['active_skill'] = sk.name
        self.working['related_sop'] = f'skills/{sk.name}/SKILL.md'
        prompt = f'[SKILL ACTIVATED: {sk.name}]\n' + build_skill_user_input(sk, skill_args)
        print(f'[Info] Skill activated: {sk.name}')
        return StepOutcome({'status': 'skill_activated', 'skill': name},
                           next_prompt=prompt)

    # ── 每轮结束：记录摘要、注入提醒 ──
    def turn_end_callback(self, response, tool_calls, tool_results, turn,
                          next_prompt, exit_reason):
        text = re.sub(r'```.*?```|<thinking>.*?</thinking>', '',
                      response.content or '', flags=re.DOTALL)
        m = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
        if m:
            summary = m.group(1).strip()
        else:
            tc = tool_calls[0] if tool_calls else {'tool_name': 'no_tool', 'args': {}}
            clean = {k: v for k, v in tc['args'].items() if not k.startswith('_')}
            summary = (f"调用工具 {tc['tool_name']}, args: {clean}"
                       if tc['tool_name'] != 'no_tool' else '直接回答了用户')
            next_prompt += '\n[TIPS] 请在回复结尾用 <summary>极简单行</summary> 总结本轮行动。'
        self.history_info.append(f'[Agent] {smart_format(summary, 100)}')

        if turn % 7 == 0 and turn > 0:
            next_prompt += (f'\n\n[DANGER] 已连续执行 {turn} 轮。禁止无效重试：'
                            '1) 探测物理边界 2) 必要时 ask_user 3) update_working_checkpoint 保存关键上下文。')
        return next_prompt
