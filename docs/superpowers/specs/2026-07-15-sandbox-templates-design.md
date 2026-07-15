# Sandbox Templates: pai-rec and turbo-x

**Date:** 2026-07-15
**Status:** Approved

## Problem

The sandbox image is a singleton. `sandbox/Dockerfile` bakes a pai-rec source
snapshot into `/opt/code` and installs the PAI recommendation plugins, and
`providers[]` carries exactly one `sandbox.default` with one `template_name`.
There is no way to run a second sandbox with different repositories, a different
toolchain, or a different network placement, and no way to bind an Agent to one.

We want a second template, `turbo-x`, whose `/opt/code` holds
`gitlab.alibaba-inc.com/PAI/image-metadata` and
`gitlab.alibaba-inc.com/pai-ee/pai-wiki` as **live git working copies** the agent
can check out branches in, deployed on a cluster with intranet access.

## Decisions

| Question | Decision |
| --- | --- |
| Binding mechanism | Named `templates` map on `sandbox.default`; Agent picks by key |
| `templates` value shape | Object (`name`, `code_writable`, `env_refs`), not a bare string |
| Legacy `template_name` | **Removed outright**; existing deployments reconfigure |
| turbo-x code acquisition | `git clone` at build time (not an OSS tarball) |
| Clone depth | `--depth 1 --no-single-branch` |
| Runtime git credentials | `GITLAB_TOKEN` injected per-template via `env_refs` |
| Dockerfile layout | Shared `base` + two thin per-template Dockerfiles |
| base cut line | Mounts, bootstrap, `AGENT_*`, ripgrep/jq/yq/git, aliyun CLI **binary** |
| Per-template | apt extras, aliyun plugins, code layer |
| `code_writable` home | The `templates` map (an image fact), not a per-agent flag |
| Unknown template key | Hard failure listing valid keys; **never** fall back |

## Two constraints discovered in the code

These invalidated parts of the original approach and shape the design.

### The platform ignores `envs`

`sandbox_providers.py:430` sends `"envs": env_contract` on create, but:

> `NOTE: AgentRun's CreateSandbox input has no 'envs' field — this is ignored by
> the platform and kept only for forward-compat.`

The contract actually arrives via `_bootstrap_env_async()`, which runs one
command **after** create that writes `export K=V` lines into `~/.bash_env`
(sourced by every non-interactive shell through the image's `BASH_ENV` hook),
base64-wrapped so no secret appears in the logged command string.

Consequences for `GITLAB_TOKEN`:

- It works. The shell tool runs non-interactive bash, so git and its credential
  helper subprocess inherit the exported variable.
- The token sits in plaintext in the sandbox's `~/.bash_env`. Sandboxes are
  per-instance and ephemeral, so this is accepted.
- Injection is a **second, best-effort call** — a failure logs and continues.
  A sandbox can therefore start without its token, surfacing only as a 401 on
  the agent's first `git fetch`.

### The session cache is already agent-scoped

`sandbox_providers.py:939` appends `agent:{agent_id}` to the scope key. Two
Agents never share a cached sandbox session, so a turbo-x Agent cannot be handed
a warm pai-rec sandbox. No change needed.

## Configuration

```yaml
providers:
  - id: sandbox.default
    settings:
      provider: agentrun_rest
      account_id_env: AGENTRUN_ACCOUNT_ID
      api_key_env: AGENTRUN_SANDBOX_API_KEY
      templates:
        pairec:
          name: sandbox-code-feiyue         # console: public network
        turbox:
          name: sandbox-turbox-feiyue       # console: VPC + vswitch/security group
          code_writable: true
          env_refs:
            GITLAB_TOKEN: GITLAB_TOKEN      # value = which env var to read
      default_template: pairec

agents:
  - id: main
    code: { enabled: true, manifest: "..." }
  - id: turbox-helper
    sandbox: { template: turbox }
    code: { enabled: true, manifest: "image-metadata / pai-wiki ..." }
```

New `AgentSandboxConfig { template: str = "" }` on `AgentProfile.sandbox`,
alongside the existing `AgentProfile.code`.

`template_name` is deleted from `ScopedSandboxProvider.__init__`,
`make_sandbox_provider`, `DEFAULT_DOCUMENT`, and the `apply_runtime_status`
grading. `templates` being non-empty replaces it in every gate.

Network placement differs per template (pai-rec public, turbo-x VPC) but is
configured **entirely in the AgentRun console**. Both templates live under the
same account and region, so endpoint and credentials stay provider-level and the
code never sees the difference.

## Runtime resolution

The provider is a singleton built from global config; the template is per-Agent.
Resolution follows the existing `default_kb_ids` pattern — the builder threads
the decision through `ToolScope.metadata`, the tool reads it back at call time.

```python
# builder.py, beside metadata["default_kb_ids"]
metadata["sandbox_template"] = agent.sandbox.template     # key only

# scope.py, beside scope_default_kb_ids()
def scope_sandbox_template() -> str:
    return str(get_current_tool_scope().metadata.get("sandbox_template") or "")

# sandbox_providers.py — store the map at construction, resolve at create
def __init__(self, settings):
    self.templates = settings.get("templates") or {}
    self.default_template = str(settings.get("default_template") or "")

def _resolve_template(self):
    key = scope_sandbox_template() or self.default_template
    tpl = self.templates.get(key)
    if tpl is None:
        raise RuntimeError(
            f"sandbox template {key!r} is not defined; "
            f"valid templates: {sorted(self.templates)}"
        )
    return key, tpl
```

Only the **key** goes into metadata. Resolving `env_refs` reads `os.environ`, and
that stays inside the provider so secrets never cross the builder or the scope.

On create (`_create_sandbox_async:428`):

```python
key, tpl = self._resolve_template()
env_contract = _build_env_contract(self, scope, scope_key)
env_contract.update({
    var: os.environ[env_name]
    for var, env_name in (tpl.get("env_refs") or {}).items()
    if os.environ.get(env_name)
})
payload = _compact_dict({"templateName": tpl["name"], ...})
```

`env_refs` merges into the contract and rides the `~/.bash_env` path above — no
new mechanism. pai-rec declares no `env_refs`, so its sandboxes never receive
`GITLAB_TOKEN`.

An `env_ref` whose variable is unset is skipped, and logs a warning naming the
variable. It cannot fail the create: the same best-effort injection path means a
token-less sandbox is already reachable, and refusing to start would take away a
sandbox that is still useful for everything except `git fetch`. The warning is
what turns the eventual 401 from a mystery into a lookup.

Unknown keys fail hard. Silently falling back would hand a turbo-x Agent a
sandbox whose `/opt/code` is full of recommendation-system source while its
manifest promises `pai-wiki` — the agent stays confused indefinitely, which is
far harder to diagnose than an error at startup.

Also updated: the SDK variant `AgentRunSdkSandboxProvider._create_sandbox:701`
(same `_resolve_template()`; it is sync but reads the same contextvar), and the
create-failure message at `:450`, which must report the resolved template name
*and* the key so an operator can see which Agent bound wrong.

## Image layering

```
sandbox/
  base/Dockerfile              # contract + toolchain
  base/agent-sandbox-bootstrap # moved, contents unchanged
  pairec/Dockerfile            # tarball code layer + PAI plugins
  turbox/Dockerfile            # git clone code layer + pai-dsw plugin
  build.sh                     # build.sh <template>: base, then template
  README.md
```

### `base/Dockerfile`

Today's Dockerfile minus the code layer and the PAI plugins:

```dockerfile
ARG BASE_IMAGE=registry.aliyuncs.com/agentrun/sandbox-code-interpreter:latest
FROM ${BASE_IMAGE}

USER root
RUN mkdir -p /mnt/system /mnt/skills /mnt/user /opt/code /home/user \
 && chown -R 1000:1000 /mnt/user /home/user

RUN apt-get update \
 && apt-get install -y --no-install-recommends ripgrep jq git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* && rg --version && jq --version && git --version

ARG YQ_VERSION=v4.44.3
RUN curl -fsSL "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64" \
      -o /usr/local/bin/yq && chmod 0755 /usr/local/bin/yq && yq --version

ARG ALIYUN_CLI_URL=https://easyrec.oss-cn-beijing.aliyuncs.com/aliyun-cli-spec/aliyun-cli-linux-latest-amd64.tgz
RUN ...   # binary only, no plugins

# Secret-free credential helper: `helper = env` makes git exec git-credential-env from PATH.
RUN printf '%s\n' '#!/bin/sh' 'echo "username=oauth2"' 'echo "password=${GITLAB_TOKEN}"' \
      > /usr/local/bin/git-credential-env \
 && chmod 0755 /usr/local/bin/git-credential-env

COPY agent-sandbox-bootstrap /usr/local/bin/agent-sandbox-bootstrap
RUN chmod +x /usr/local/bin/agent-sandbox-bootstrap

ENV AGENT_SYSTEM_PATH=/mnt/system ... HOME=/home/user
USER 1000
ENTRYPOINT ["agent-sandbox-bootstrap"]
CMD ["/usr/local/bin/entrypoint.sh", "process-compose", "up", "--tui=false", "--no-server"]
```

The helper script lives in base (generic, secret-free); the gitlab host binding
lives in turbox.

### `pairec/Dockerfile`

`PAIREC_CODE_ARCHIVE_URL` becomes `CODE_ARCHIVE_URL` — the prefix is redundant
inside `sandbox/pairec/`.

```dockerfile
FROM sandbox-base:latest
ARG CODE_ARCHIVE_URL=https://pai-rag.oss-cn-hangzhou.aliyuncs.com/production_artifacts/pairec_code/pairec_code_20260623.tar.gz
USER root
RUN if [ -n "${CODE_ARCHIVE_URL}" ]; then \
      curl -fsSL "${CODE_ARCHIVE_URL}" -o /tmp/c.tar.gz \
      && tar -xzf /tmp/c.tar.gz --strip-components=1 -C /opt/code \
      && rm -f /tmp/c.tar.gz && chmod -R a+rX /opt/code; \
    fi
USER 1000
RUN aliyun plugin install --names aliyun-cli-eas aliyun-cli-pairecservice \
      aliyun-cli-pai-dsw aliyun-cli-paifeaturestore && aliyun plugin list
```

### `turbox/Dockerfile`

```dockerfile
FROM sandbox-base:latest
ARG TURBOX_REPOS="\
  http://gitlab.alibaba-inc.com/PAI/image-metadata.git@master \
  http://gitlab.alibaba-inc.com/pai-ee/pai-wiki.git@master"
USER root
RUN --mount=type=secret,id=gitcred,target=/root/.git-credentials \
    for spec in ${TURBOX_REPOS}; do \
      url="${spec%@*}"; ref="${spec##*@}"; name="$(basename "$url" .git)"; \
      git -c credential.helper='store --file=/root/.git-credentials' \
          clone --depth 1 --no-single-branch -b "$ref" "$url" "/opt/code/$name"; \
    done \
 && git config --system credential."http://gitlab.alibaba-inc.com".helper env \
 && chown -R 1000:1000 /opt/code
USER 1000
RUN aliyun plugin install --names aliyun-cli-pai-dsw && aliyun plugin list
```

`--depth 1 --no-single-branch` fetches every branch tip at one commit each, so
the agent can `checkout` any branch that existed at build time **offline**, and
`fetch --deepen` / `--unshallow` for history — the cluster reaches the intranet.

**No token reaches any image layer.** Build time uses a BuildKit secret mount
with `store --file=`, which keeps the remote URL clean. Embedding the token in
the URL instead (`http://user:token@gitlab...`) would persist it in
`/opt/code/<repo>/.git/config` — and `.git` is now deliberately **retained**, so
the token would ship with the image. Runtime takes a different path entirely:
`credential.helper=env` → `git-credential-env` → `$GITLAB_TOKEN`.

`chown -R 1000:1000 /opt/code` is turbox-only. pai-rec stays root-owned `a+rX`.

base is not pushed; `build.sh <template>` builds `sandbox-base:latest` locally
first. CI must run both steps — `FROM sandbox-base:latest` fails on a clean
machine.

## `/opt/code` semantics

`soul.py:197` hardcodes read-only into the prompt:

> "A **read-only** code layer at `/opt/code` ... It is read-only reference
> material — **do not try to modify it**"

A turbo-x manifest describing checkoutable working copies would land in the same
system prompt as that sentence — a self-contradictory instruction that most
likely shows up as the model reading repos but refusing to check out, with no
trace of why. **The manifest cannot fix this; `soul.py` must branch.**

`code_writable` lives in the `templates` map because writability is a **property
of the image** (turbox's `chown -R 1000:1000 /opt/code`), not an admin
preference per Agent. Declared there, the fact sits next to its cause; declared
per-Agent, every new turbo-x Agent must remember to tick it, and a missed tick
reproduces the contradictory prompt.

The builder resolves `agent.sandbox.template` → `templates[key].code_writable` →
`render_stable_system_prompt(code_writable=...)`, and `_code_layer_block`
selects wording from it.

Three coupled details:

- **Two prompt render sites** — `builder.py:167-175` and `builder.py:315-316`
  (subagent path). Both must pass `code_writable`; missing one gives subagents
  the contradictory prompt.
- **Subagents inherit the wrong template.** `builder.py:~340` does
  `metadata = dict(parent_scope.metadata or {})`. `default_kb_ids` is explicitly
  set-or-popped there; `sandbox_template` must do the same, or a turbo-x
  subagent under a pai-rec parent silently runs on the pai-rec image.
- **Checkouts are ephemeral.** The container's per-instance writable layer is
  discarded with the sandbox, reverting to the baked tip. The turbo-x manifest
  must say so, or the agent will assume a checkout persisted.

`sandbox/README.md`'s `/opt/*` vs `/mnt/*` convention needs **rewording, not
weakening**. It currently reads "`/opt/*` = baked into the image (**read-only**
code layer)". Read-only no longer holds, but the rule was always really about
**backing store** — image layer vs NAS. Restate it as "`/opt/*` = image layer
(local disk), `/mnt/*` = NAS mounts", and its actual constraint ("do not move a
NAS mount under `/opt`, or a baked layer under `/mnt`") survives untouched.

`agent-sandbox-bootstrap` is **unchanged** and `CONTRACT_VERSION` stays `1` — it
validates only `/mnt/system|skills|user`. `/opt/code` is outside its checks and
the mount contract is unaffected.

## Blast radius

**Backend** — `sandbox_providers.py` (store map, `_resolve_template()`, both
`_create_sandbox*`, `make_sandbox_provider` gate, `:450` message), `scope.py`
(`scope_sandbox_template()`), `builder.py` (metadata thread, two render sites,
subagent set-or-pop), `agent_config.py` (`AgentSandboxConfig`, `DEFAULT_DOCUMENT`,
`:706/712` grading), `soul.py` (`_code_layer_block` branch).

**Frontend** — `SettingsView.tsx:2414-2514` (single input → templates map
editor: key / name / code_writable / env_refs), `:1083` (template picker beside
the Agent's code config), `i18n/en.ts` + `zh.ts`.

**Image** — `sandbox/{base,pairec,turbox}/Dockerfile`, `build.sh`,
`agent-sandbox-bootstrap` moved to `base/`.

**Docs** — `sandbox/README.md` (rewrite), `docs/design/skill_install_mount_dependencies.md`
(`/opt/code`), `backend/README.md:36`.

**Existing tests to update** — `test_builtin_tools.py` (12× `"template_name":
"code-template"`), `test_lean_main_boot.py:38` (asserts the `DEFAULT_DOCUMENT`
value, not migration logic — becomes `templates == {}`), `SettingsView.test.tsx`
(4 sites).

## Testing

New tests, each pinning a trap found above:

- Unknown key raises, message lists valid keys, **no** fallback.
- `env_refs` reach only the bound template's contract — assert a pai-rec sandbox's
  envs contain **no** `GITLAB_TOKEN`.
- A turbo-x subagent under a pai-rec parent resolves to turbox, not the inherited
  pai-rec.
- `code_writable=True` renders a prompt without "do not try to modify it";
  `False` keeps it.
- `make_sandbox_provider` returns `None` when `templates` is empty.
- `apply_runtime_status` grades on `templates`, not `template_name`.

Images are verified by build, not unit test: after `build.sh turbox`, `docker run`
and check `rg`/`jq`/`yq`/`git`/`aliyun` resolve, `/opt/code` is owned by 1000,
`git -C /opt/code/pai-wiki branch -r` lists branches offline, and `docker history`
plus a layer unpack find **no token**.
