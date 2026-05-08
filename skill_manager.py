"""Skill 注册 / 发现 / 触发系统。
仿 Claude Code 的 SKILL.md 声明式框架：
- 文件驱动：skills/<name>/SKILL.md 自动扫描注册
- 用户触发：/skill-name [args] 斜杠命令
- LLM 触发：use_skill 工具调用
- 系统提示注入：LLM 可见可用技能列表
"""
import os, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SkillDef:
    name: str
    description: str
    trigger: str
    instructions: str
    path: str
    allowed_tools: List[str] = field(default_factory=list)


def parse_skill_md(filepath: str) -> Optional[SkillDef]:
    """解析 SKILL.md：YAML frontmatter + Markdown body。"""
    try:
        text = open(filepath, 'r', encoding='utf-8').read()
    except Exception:
        return None

    m = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
    if not m:
        return None

    frontmatter, body = m.group(1), m.group(2).strip()
    meta = {}
    for line in frontmatter.splitlines():
        km = re.match(r'^(\S[\w-]*)\s*:\s*(.+)', line)
        if km:
            meta[km.group(1).strip()] = km.group(2).strip().strip('"').strip("'")

    dir_name = os.path.basename(os.path.dirname(filepath))
    name = meta.get('name', dir_name)
    description = meta.get('description', body.split('\n')[0] if body else name)
    trigger = meta.get('trigger', f'/{name}')
    if not trigger.startswith('/'):
        trigger = '/' + trigger

    allowed_tools = []
    if 'allowed-tools' in meta:
        allowed_tools = [t.strip() for t in meta['allowed-tools'].split(',') if t.strip()]

    return SkillDef(
        name=name,
        description=description,
        trigger=trigger,
        instructions=body,
        path=filepath,
        allowed_tools=allowed_tools,
    )


def scan_skills(base_dir: str) -> Dict[str, SkillDef]:
    """扫描 base_dir/*/SKILL.md，返回 {name: SkillDef}。"""
    skills = {}
    if not os.path.isdir(base_dir):
        return skills
    try:
        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            skill_file = os.path.join(entry.path, 'SKILL.md')
            if os.path.isfile(skill_file):
                sd = parse_skill_md(skill_file)
                if sd:
                    skills[sd.name] = sd
    except OSError:
        pass
    return skills


def get_skills_prompt(skills: Dict[str, SkillDef]) -> str:
    """生成注入系统提示词的技能列表。"""
    if not skills:
        return ''
    lines = [
        '\n## Available Skills',
        '**强约束**：当用户的请求匹配下方任一 skill 的 description（即使未使用 `/` 斜杠命令、'
        '即使只是提到 skill 名字或其领域关键词），你**必须**立即调用 `use_skill` 工具启动该 skill，'
        '然后按 skill 注入的 SOP 指引执行。',
        '禁止：① 仅回复 `<summary>` 就结束本轮；② 凭固有知识直接回答而跳过 skill；'
        '③ 多个 skill 可选时不调用任何一个。',
        '若判断多个 skill 都可能相关，选 description 与任务最贴合的那个；若不确定再 `ask_user` 澄清。',
        '',
        'Skills:',
    ]
    for sk in skills.values():
        tools_note = f' (tools: {", ".join(sk.allowed_tools)})' if sk.allowed_tools else ''
        lines.append(f'- `{sk.name}` (trigger: `{sk.trigger}`): {sk.description}{tools_note}')
    lines.append('')
    return '\n'.join(lines)


def get_use_skill_schema() -> dict:
    """返回 use_skill 工具的 OpenAI function-calling 格式定义。"""
    return {
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": "Invoke a registered skill by name. Use when user requests a skill or when a task matches a skill's description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "The skill name to invoke (see Available Skills in system prompt)",
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional arguments to pass to the skill",
                    },
                },
                "required": ["skill"],
            },
        },
    }


def match_skill(task: str, skills: Dict[str, SkillDef]) -> tuple:
    """匹配斜杠命令。返回 (SkillDef, remaining_args) 或 (None, None)。"""
    if not task.startswith('/') or not skills:
        return None, None
    for sk in skills.values():
        trigger = sk.trigger
        if task == trigger or task.startswith(trigger + ' '):
            remaining = task[len(trigger):].strip()
            return sk, remaining
    return None, None


def build_skill_user_input(skill: SkillDef, user_args: str) -> str:
    """将技能指令注入为用户消息。

    注入 skill 目录绝对路径，约定正文中的相对路径（scripts/xxx, references/xxx
    等）均相对此目录；agent 调用脚本或读取资源时应拼成绝对路径。
    """
    skill_dir = os.path.dirname(skill.path)
    return (
        f'<skill name="{skill.name}" dir="{skill_dir}">\n'
        f'[IMPORTANT] 本 skill 目录绝对路径：{skill_dir}\n'
        f'正文中的相对路径（scripts/xxx、references/xxx 等）均相对此目录。\n'
        f'调用脚本或读取资源前，请拼成绝对路径再操作。\n\n'
        f'[WORKSPACE SAFETY] 调用 code_run / shell 工具时，优先省略 cwd 或使用 "."，让后端使用当前用户 session workspace；不要写死项目源码根目录，也不要将 cwd 设置为本 skill 目录。\n'
        f'如需执行 skill 中的脚本，请保持 cwd 不变，并在命令里使用脚本的绝对路径。\n'
        f'如需创建临时文件，请写入 workspace 下的 ./.tmp/ 或相对路径，不要使用 /tmp、/var/tmp 或 workspace 外的绝对路径。\n\n'
        f'{skill.instructions}\n'
        f'</skill>\n\n'
        f'User request: {user_args or "Execute the skill."}'
    )
