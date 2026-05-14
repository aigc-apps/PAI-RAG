"""Background long-term memory review after a foreground run completes."""
import contextlib
import copy
import io
import logging
import os
import threading
from contextlib import contextmanager

from agent_loop import StepOutcome, agent_runner_loop
from backend.celery_app import celery_app
from backend.tool_schemas import background_memory_review_tools_schema
from llm_client import LLMClient
from session_store import SERVER_USER_ID
from tools import GenericHandler, file_read
import settings as config
import runtime_config


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


def background_review_enabled():
    return bool(getattr(config, 'BACKGROUND_MEMORY_REVIEW_ENABLED', True))


def _thread_lock(memory_root):
    key = os.path.abspath(memory_root)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


@contextmanager
def memory_review_lock(memory_root):
    os.makedirs(memory_root, exist_ok=True)
    lock_path = os.path.join(memory_root, '.memory_review.lock')
    with _thread_lock(memory_root):
        with open(lock_path, 'a+', encoding='utf-8') as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_text(path, default='(empty)'):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except OSError:
        return default


def build_background_review_system_prompt(memory_root, user_id=SERVER_USER_ID):
    index = _read_text(os.path.join(memory_root, 'global_index.txt'))
    return (
        '# Role: 后台长期记忆审查 Agent\n'
        '你在用户任务完成后异步运行。主任务最终答案已经交付；你的唯一职责是判断是否需要沉淀长期记忆。\n\n'
        '规则：\n'
        '- 若本次任务没有长期可复用信息，直接输出 `<summary>无需沉淀</summary>` 并结束。\n'
        '- 若发现行动验证成功且长期有效的环境事实、用户偏好、避坑点或可复用流程，调用 `start_long_term_update`。\n'
        '- 调用 `start_long_term_update` 后，严格按工具返回的 SOP 使用 file_read / file_patch / file_write 更新记忆。\n'
        '- 不保存当前任务进度、一次性结论、临时 TODO、未经验证的推测、通用常识或任何密钥/token/password/AK/SK。\n'
        '- 不修改用户 workspace，不向用户继续提问，不输出新的业务诊断报告。\n\n'
        f'[MEMORY SCOPE] user_id={user_id}; memory_root={os.path.abspath(memory_root)}\n\n'
        '## L1 索引\n'
        f'{index}'
    )


def build_background_review_user_prompt(session_id='', run_id='', task_text=''):
    return (
        '请审查上方完整会话快照，执行一次长期记忆沉淀评估。\n'
        f'- session_id: {session_id}\n'
        f'- run_id: {run_id}\n'
        f'- task: {str(task_text or "")[:1000]}\n\n'
        '先从严判断是否有值得跨会话复用的经验。若有，调用 `start_long_term_update` 并完成必要的最小化记忆更新；'
        '若没有，直接 `<summary>无需沉淀</summary>`。'
    )


def _new_review_client(history):
    client = LLMClient(
        api_key=config.API_KEY,
        api_base=getattr(config, 'API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        model=runtime_config.get_active_model(),
        max_tokens=getattr(config, 'MAX_TOKENS', 8192),
        history_trim_tokens=getattr(config, 'HISTORY_TRIM_TOKENS', 80000),
        timeout=getattr(config, 'TIMEOUT', 300),
    )
    client.history = copy.deepcopy(history or [])
    return client


class BackgroundReviewHandler(GenericHandler):
    def _anchor_prompt(self, skip=False):
        return '\n'

    def do_file_read(self, args, response):
        path = self._abs(args.get('path', ''))
        result = file_read(
            path,
            start=args.get('start', 1),
            keyword=args.get('keyword'),
            count=args.get('count', 200),
            show_linenos=args.get('show_linenos', True),
        )
        if args.get('show_linenos', True) and not result.startswith('Error:'):
            result = '由于设置了 show_linenos，以下返回信息为：(行号|)内容\n' + result
        return StepOutcome(result, next_prompt='\n')


def _new_review_handler(memory_root, active_skill=''):
    memory_root = os.path.abspath(memory_root)
    handler = BackgroundReviewHandler(
        memory_root,
        ROOT,
        workspace_root=memory_root,
        readonly_roots=[],
        writable_roots=[memory_root],
        memory_root=memory_root,
        long_term_memory_enabled=True,
    )
    if active_skill:
        handler.working['active_skill'] = active_skill
        handler.working['related_sop'] = f'skills/{active_skill}/SKILL.md'
    return handler


def run_background_memory_review(
    *,
    user_id=SERVER_USER_ID,
    session_id='',
    run_id='',
    task_text='',
    llm_history=None,
    memory_root='',
    active_skill='',
):
    if not memory_root:
        return {'status': 'skipped', 'reason': 'missing_memory_root'}

    memory_root = os.path.abspath(memory_root)
    with memory_review_lock(memory_root):
        client = _new_review_client(llm_history)
        handler = _new_review_handler(memory_root, active_skill=active_skill)
        system_prompt = build_background_review_system_prompt(memory_root, user_id=user_id)
        user_prompt = build_background_review_user_prompt(session_id, run_id, task_text)
        max_turns = int(getattr(config, 'BACKGROUND_MEMORY_REVIEW_MAX_TURNS', 8) or 8)
        logger.info('Starting background memory review: session=%s run=%s', session_id, run_id)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exit_reason = agent_runner_loop(
                client=client,
                system_prompt=system_prompt,
                user_input=user_prompt,
                handler=handler,
                tools_schema=background_memory_review_tools_schema(),
                max_turns=max_turns,
            )
        logger.info(
            'Background memory review finished: session=%s run=%s result=%s',
            session_id,
            run_id,
            (exit_reason or {}).get('result'),
        )
        return {'status': 'completed', 'exit_reason': exit_reason}


def _run_background_memory_review_safe(payload):
    try:
        return run_background_memory_review(**payload)
    except Exception as exc:
        logger.warning('Background memory review failed: %s', exc, exc_info=True)
        return {'status': 'error', 'error': str(exc)}


def schedule_background_memory_review(
    *,
    user_id=SERVER_USER_ID,
    session_id='',
    run_id='',
    task_text='',
    llm_history=None,
    memory_root='',
    active_skill='',
    final_status='completed',
    long_term_enabled=False,
    use_celery=False,
):
    if not background_review_enabled():
        logger.info(
            'Background memory review skipped: reason=disabled session=%s run=%s',
            session_id,
            run_id,
        )
        return False
    if final_status != 'completed':
        logger.info(
            'Background memory review skipped: reason=final_status status=%s session=%s run=%s',
            final_status,
            session_id,
            run_id,
        )
        return False
    if not long_term_enabled:
        logger.info(
            'Background memory review skipped: reason=long_term_disabled session=%s run=%s',
            session_id,
            run_id,
        )
        return False
    if not memory_root:
        logger.info(
            'Background memory review skipped: reason=missing_memory_root session=%s run=%s',
            session_id,
            run_id,
        )
        return False
    payload = {
        'user_id': user_id or SERVER_USER_ID,
        'session_id': session_id,
        'run_id': run_id,
        'task_text': task_text,
        'llm_history': copy.deepcopy(llm_history or []),
        'memory_root': memory_root,
        'active_skill': active_skill or '',
    }
    if use_celery:
        try:
            review_memory_task.delay(payload)
        except Exception as exc:
            logger.warning(
                'Background memory review enqueue failed: session=%s run=%s error=%s',
                session_id,
                run_id,
                exc,
                exc_info=True,
            )
            raise
        logger.info(
            'Background memory review enqueued: backend=celery session=%s run=%s memory_root=%s',
            session_id,
            run_id,
            os.path.abspath(memory_root),
        )
        return True

    thread = threading.Thread(
        target=_run_background_memory_review_safe,
        args=(payload,),
        daemon=True,
        name=f'memory-review-{run_id or session_id or "run"}',
    )
    thread.start()
    logger.info(
        'Background memory review scheduled: backend=thread session=%s run=%s memory_root=%s',
        session_id,
        run_id,
        os.path.abspath(memory_root),
    )
    return True


@celery_app.task(name='pai_rag.review_memory')
def review_memory_task(payload):
    return _run_background_memory_review_safe(payload or {})
