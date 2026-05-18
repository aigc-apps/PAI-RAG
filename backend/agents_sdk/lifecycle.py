"""Run/session lifecycle constants — source of truth.

Owns both the session-level statuses (``SESSION_*``) and the SDK-runner
``RunState`` statuses (``RUN_STATE_*``). ``backend.agent_service`` now
imports these back; previous direction was the reverse.
"""

SESSION_IDLE = 'idle'
SESSION_RUNNING = 'running'
SESSION_WAITING_USER = 'waiting_user'
SESSION_COMPLETED = 'completed'
SESSION_FAILED = 'failed'
SESSION_CANCELLED = 'cancelled'
ACTIVE_SESSION_STATUSES = {SESSION_RUNNING, SESSION_WAITING_USER}

RUN_STATE_RUNNING = 'running'
RUN_STATE_REQUIRES_ACTION = 'requires_action'
RUN_STATE_COMPLETED = 'completed'
RUN_STATE_FAILED = 'failed'
RUN_STATE_CANCELLED = 'cancelled'
RUN_STATE_EXPIRED = 'expired'

RUN_STATE_TERMINAL = frozenset({
    RUN_STATE_COMPLETED,
    RUN_STATE_FAILED,
    RUN_STATE_CANCELLED,
    RUN_STATE_EXPIRED,
})
RUN_STATE_ACTIVE = frozenset({RUN_STATE_RUNNING, RUN_STATE_REQUIRES_ACTION})

__all__ = [
    'ACTIVE_SESSION_STATUSES',
    'SESSION_CANCELLED',
    'SESSION_COMPLETED',
    'SESSION_FAILED',
    'SESSION_IDLE',
    'SESSION_RUNNING',
    'SESSION_WAITING_USER',
    'RUN_STATE_RUNNING',
    'RUN_STATE_REQUIRES_ACTION',
    'RUN_STATE_COMPLETED',
    'RUN_STATE_FAILED',
    'RUN_STATE_CANCELLED',
    'RUN_STATE_EXPIRED',
    'RUN_STATE_TERMINAL',
    'RUN_STATE_ACTIVE',
]
