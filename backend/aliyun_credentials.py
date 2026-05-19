"""Per-run Aliyun CLI credential profile management."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - this service runs on Linux in prod/CI.
    fcntl = None


DEFAULT_REGION_ID = 'cn-beijing'
RUNTIME_PROFILE_PREFIX = 'pai-rag-runtime-'
REQUEST_CREDENTIALS_ENV = 'PAI_RAG_REQUEST_ALIYUN_CREDENTIALS'
ALIYUN_CONFIGURE_TIMEOUT_SECONDS = 15


class AliyunCredentialError(ValueError):
    """The request-supplied credential payload is invalid."""


class AliyunConfigError(RuntimeError):
    """The local Aliyun CLI config file cannot be updated safely."""


@dataclass(frozen=True)
class AliyunCredentials:
    access_key_id: str
    access_key_secret: str
    region_id: str = DEFAULT_REGION_ID


@dataclass(frozen=True)
class AliyunProfileLease:
    profile_name: str
    config_path: Path

    @property
    def env(self) -> dict[str, str]:
        return {
            'ALIBABA_CLOUD_PROFILE': self.profile_name,
            REQUEST_CREDENTIALS_ENV: '1',
        }

    def cleanup(self) -> bool:
        return remove_profile(self.profile_name, config_path=self.config_path)


def pop_aliyun_credentials(body: dict[str, Any]) -> AliyunCredentials | None:
    """Remove and validate the optional ``aliyun_credentials`` request field."""
    if 'aliyun_credentials' not in body:
        return None
    return parse_aliyun_credentials(body.pop('aliyun_credentials'))


def parse_aliyun_credentials(raw: Any) -> AliyunCredentials:
    if not isinstance(raw, dict):
        raise AliyunCredentialError('aliyun_credentials must be an object')

    access_key_id = _required_string(raw, 'access_key_id')
    access_key_secret = _required_string(raw, 'access_key_secret')
    region_id = _optional_string(raw, 'region_id') or DEFAULT_REGION_ID
    return AliyunCredentials(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        region_id=region_id,
    )


def write_temporary_profile(
    credentials: AliyunCredentials,
    *,
    config_path: str | os.PathLike[str] | None = None,
    configure_with_cli: bool | None = None,
) -> AliyunProfileLease:
    path = _config_path(config_path)
    profile_name = RUNTIME_PROFILE_PREFIX + uuid.uuid4().hex
    if configure_with_cli is None:
        configure_with_cli = config_path is None
    should_run_cli_configure = configure_with_cli and _cli_home_for_config(path) is not None
    profile = {
        'name': profile_name,
        'mode': 'AK',
        'access_key_id': credentials.access_key_id,
        'access_key_secret': credentials.access_key_secret,
        'region_id': credentials.region_id,
        'output_format': 'json',
        'language': 'en',
    }

    with _locked_config(path):
        data = _read_config(path)
        if should_run_cli_configure:
            _run_configure_set_best_effort(credentials, profile_name, path)
            data = _read_config(path)
        profiles = _profiles(data)
        data['profiles'] = [item for item in profiles if item.get('name') != profile_name]
        data['profiles'].append(profile)
        data['current'] = profile_name
        _write_config(path, data)

    return AliyunProfileLease(profile_name=profile_name, config_path=path)


def remove_profile(
    profile_name: str,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> bool:
    if not profile_name:
        return False
    path = _config_path(config_path)
    if not path.exists():
        return False

    with _locked_config(path):
        data = _read_config(path)
        profiles = _profiles(data)
        kept = [item for item in profiles if item.get('name') != profile_name]
        if len(kept) == len(profiles):
            return False
        data['profiles'] = kept
        if data.get('current') == profile_name:
            data['current'] = _fallback_current_profile(kept)
        _write_config(path, data)
        return True


def _required_string(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AliyunCredentialError(f'aliyun_credentials.{field} must be a non-empty string')
    return value.strip()


def _optional_string(raw: dict[str, Any], field: str) -> str:
    if field not in raw or raw.get(field) is None:
        return ''
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AliyunCredentialError(f'aliyun_credentials.{field} must be a non-empty string when provided')
    return value.strip()


def _config_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return Path.home() / '.aliyun' / 'config.json'


def _cli_home_for_config(path: Path) -> Path | None:
    if path.name != 'config.json' or path.parent.name != '.aliyun':
        return None
    return path.parent.parent


def _run_configure_set_best_effort(
    credentials: AliyunCredentials,
    profile_name: str,
    path: Path,
) -> None:
    home = _cli_home_for_config(path)
    if home is None:
        return
    env = os.environ.copy()
    env['HOME'] = str(home)
    cmd = [
        'aliyun',
        'configure',
        'set',
        '--profile',
        profile_name,
        '--mode',
        'AK',
        '--region',
        credentials.region_id,
        '--access-key-id',
        credentials.access_key_id,
        '--access-key-secret',
        credentials.access_key_secret,
    ]
    try:
        subprocess.run(
            cmd,
            cwd=str(home),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=ALIYUN_CONFIGURE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # The direct config writer below is the authoritative fallback. Do not
        # include command args here because they contain request credentials.
        return


@contextmanager
def _locked_config(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _chmod_best_effort(path.parent, 0o700)
    lock_path = path.parent / 'config.json.lock'
    with open(lock_path, 'a+', encoding='utf-8') as lock_file:
        _chmod_best_effort(lock_path, 0o600)
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'current': 'default', 'profiles': []}
    try:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
    except OSError as exc:
        raise AliyunConfigError(f'Failed to read Aliyun CLI config: {path}') from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AliyunConfigError(f'Aliyun CLI config is invalid JSON: {path}') from exc
    if not isinstance(data, dict):
        raise AliyunConfigError(f'Aliyun CLI config must be a JSON object: {path}')
    if 'profiles' not in data:
        data['profiles'] = []
    _profiles(data)
    return data


def _profiles(data: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = data.get('profiles')
    if not isinstance(profiles, list):
        raise AliyunConfigError('Aliyun CLI config field "profiles" must be an array')
    for item in profiles:
        if not isinstance(item, dict):
            raise AliyunConfigError('Aliyun CLI config profiles must be objects')
    return profiles


def _write_config(path: Path, data: dict[str, Any]) -> None:
    tmp_name = ''
    try:
        with tempfile.NamedTemporaryFile(
            'w',
            encoding='utf-8',
            dir=path.parent,
            prefix='config.',
            suffix='.tmp',
            delete=False,
        ) as file:
            tmp_name = file.name
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
        tmp_name = ''
        _chmod_best_effort(path, 0o600)
    except OSError as exc:
        raise AliyunConfigError(f'Failed to write Aliyun CLI config: {path}') from exc
    finally:
        if tmp_name:
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def _fallback_current_profile(profiles: list[dict[str, Any]]) -> str:
    names = [str(item.get('name') or '') for item in profiles]
    if 'default' in names:
        return 'default'
    return next((name for name in names if name), 'default')


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass
