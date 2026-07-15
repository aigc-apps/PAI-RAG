from agent.soul import render_stable_system_prompt, render_subagent_system_prompt

MANIFEST = "- image-metadata: PAI image metadata\n- pai-wiki: internal docs"
TOOLS = ["shell", "code_interpreter"]


def _prompt(writable: bool, render=render_stable_system_prompt) -> str:
    return render(
        "You are a helper.", tool_names=TOOLS,
        code_enabled=True, code_manifest=MANIFEST, code_writable=writable,
    )


def test_read_only_prompt_forbids_modification():
    prompt = _prompt(False)
    assert "do not try to modify it" in prompt
    assert "read-only code layer" in prompt


def test_writable_prompt_does_not_forbid_modification():
    prompt = _prompt(True)
    assert "do not try to modify it" not in prompt
    assert "read-only reference material" not in prompt


def test_writable_prompt_describes_git_working_copies():
    prompt = _prompt(True)
    assert "git working copies" in prompt
    assert "check out" in prompt
    assert "discarded" in prompt  # ephemerality must be stated


def test_writable_flag_reaches_subagent_prompt():
    prompt = _prompt(True, render=render_subagent_system_prompt)
    assert "do not try to modify it" not in prompt
    assert "git working copies" in prompt


def test_manifest_still_leads_the_block():
    assert MANIFEST in _prompt(True)
    assert MANIFEST in _prompt(False)
