# tests/app/test_lean_import_isolation.py
import os, sys, subprocess


def test_lean_app_imports_without_heavy_deps():
    """The lean service must import with the heavy ML/trace stack BLOCKED."""
    code = (
        "import builtins\n"
        "BLOCKED = ('torch','transformers','llama_index','tokenizers','accelerate',"
        "'safetensors','extensions.trace','opentelemetry','openinference')\n"
        "_real = builtins.__import__\n"
        "def guard(name,*a,**k):\n"
        "    base = name.split('.')[0]\n"
        "    for b in BLOCKED:\n"
        "        if name == b or name.startswith(b + '.') or base == b:\n"
        "            raise ImportError('blocked heavy dep: ' + name)\n"
        "    return _real(name,*a,**k)\n"
        "builtins.__import__ = guard\n"
        "import app.lean_main\n"
        "print('LEAN_OK')\n"
    )
    backend = os.path.join(os.path.dirname(__file__), "../../backend")
    r = subprocess.run([sys.executable, "-c", code], cwd=backend,
                       capture_output=True, text=True)
    assert "LEAN_OK" in r.stdout, f"lean import pulled a heavy dep:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
