"""User authentication helpers for the HTTP backend."""
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt


USERNAME_RE = re.compile(r'^[A-Za-z0-9_.-]{3,32}$')
BCRYPT_MAX_PASSWORD_BYTES = 72


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    username: str
    is_service: bool = False


class UserStore:
    def __init__(self, db_path):
        self._db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=30000')
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                '''
            )
            conn.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')

    @staticmethod
    def normalize_username(username):
        return (username or '').strip().lower()

    def create_user(self, username, password):
        username = self.normalize_username(username)
        if not USERNAME_RE.fullmatch(username):
            raise ValueError('Username must be 3-32 characters: letters, numbers, dot, dash, underscore')
        if not password or len(password) < 6:
            raise ValueError('Password must be at least 6 characters')
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > BCRYPT_MAX_PASSWORD_BYTES:
            raise ValueError('Password must be at most 72 bytes')

        now = datetime.now(timezone.utc).isoformat()
        user = {
            'user_id': str(uuid.uuid4()),
            'username': username,
            'password_hash': bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode('utf-8'),
            'created_at': now,
            'updated_at': now,
        }
        try:
            with self._connect() as conn:
                conn.execute(
                    '''
                    INSERT INTO users (user_id, username, password_hash, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (user['user_id'], user['username'], user['password_hash'], user['created_at'], user['updated_at']),
                )
        except sqlite3.IntegrityError as e:
            raise ValueError('Username already exists') from e
        return self.public_user(user)

    def authenticate(self, username, password):
        row = self.get_by_username(username)
        if row is None:
            return None
        password_bytes = (password or '').encode('utf-8')
        if len(password_bytes) > BCRYPT_MAX_PASSWORD_BYTES:
            return None
        if not bcrypt.checkpw(password_bytes, row['password_hash'].encode('utf-8')):
            return None
        return self.public_user(row)

    def get_by_id(self, user_id):
        with self._connect() as conn:
            row = conn.execute(
                '''
                SELECT user_id, username, password_hash, created_at, updated_at
                FROM users
                WHERE user_id = ?
                ''',
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_by_username(self, username):
        username = self.normalize_username(username)
        with self._connect() as conn:
            row = conn.execute(
                '''
                SELECT user_id, username, password_hash, created_at, updated_at
                FROM users
                WHERE username = ?
                ''',
                (username,),
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def public_user(user):
        return {
            'user_id': user['user_id'],
            'username': user['username'],
            'created_at': user.get('created_at', ''),
            'updated_at': user.get('updated_at', ''),
        }


def create_token(user, secret, ttl_seconds):
    now = datetime.now(timezone.utc)
    payload = {
        'sub': user['user_id'],
        'username': user['username'],
        'iat': now,
        'exp': now + timedelta(seconds=ttl_seconds),
    }
    return jwt.encode(payload, secret, algorithm='HS256')


def decode_token(token, secret):
    payload = jwt.decode(token, secret, algorithms=['HS256'])
    return {
        'user_id': payload.get('sub') or '',
        'username': payload.get('username') or '',
    }
