# syntax=docker/dockerfile:1
# =============================================================================
# turbo-x sandbox template. Clones the intranet gitlab repositories into
# /opt/code as live git working copies the agent can check out branches in.
#
#   sandbox/build.sh turbox      # needs GITLAB_TOKEN and intranet access
#
# Register in AgentRun as a code-interpreter template with network mode VPC
# (vswitch + security group reaching gitlab.alibaba-inc.com), under the same
# account_id + region as pai-rec, and point settings.templates.turbox.name at it.
# settings.templates.turbox must set code_writable: true — the prompt's wording
# is driven by it, and a mismatch tells the model not to touch repositories its
# own manifest says it may check out.
# =============================================================================

FROM sandbox-base:latest

ARG TURBOX_REPOS="\
  http://gitlab.alibaba-inc.com/PAI/image-metadata.git@master \
  http://gitlab.alibaba-inc.com/pai-ee/pai-wiki.git@master"

# --depth 1 --no-single-branch fetches every branch tip at one commit each, so the
# agent can check out any branch that existed at build time offline, and
# `git fetch --deepen`/`--unshallow` for history (the VPC template reaches gitlab).
#
# The token arrives as a BuildKit secret and is consumed via `store --file=`, so
# the remote URL stays clean. Do NOT embed it in the URL
# (http://user:token@gitlab...): .git is deliberately RETAINED here, so the token
# would persist in /opt/code/<repo>/.git/config and ship with the image.
#
# chown to 1000 is what makes checkout possible, and is why this template sets
# code_writable. Writes land in the container's per-instance layer and are
# discarded with the sandbox — the image and upstream are never touched.
USER root
# The secret mount above is READ-ONLY, but `git-credential-store` writes the
# entry back to its file after a successful auth -- pointed straight at the
# mount, that write fails with "unable to get credential storage lock" on
# every clone. The clone still succeeds (git carries on past a credential
# helper failure), but the build log fills with `fatal:` lines that read
# exactly like the real failure a user watching the build would be looking
# for. Copy the secret to a writable path first and point the helper there
# instead; the copy must not survive this RUN (one layer, created and
# deleted in the same instruction) or it would ship inside the image.
#
# Every step below is `;`-separated, NOT `&&`-joined. `set -eu` is suppressed
# inside any command of an AND-OR list except the last, and that suppression
# reaches into a loop body -- so `done && rm ...` would let a failed clone pass
# silently and the RUN would exit 0 with a repository missing from /opt/code,
# printing a green "cloned code layer:" on the way out. With `;` a failed clone
# aborts the build. (A failed RUN commits no layer, so skipping the `rm` on that
# path leaks nothing.)
RUN --mount=type=secret,id=gitcred,target=/root/.git-credentials \
    set -eu; \
    cp /root/.git-credentials /root/.git-credentials.rw; \
    for spec in ${TURBOX_REPOS}; do \
      url="${spec%@*}"; ref="${spec##*@}"; name="$(basename "$url" .git)"; \
      git -c credential.helper='store --file=/root/.git-credentials.rw' \
          clone --depth 1 --no-single-branch -b "$ref" "$url" "/opt/code/$name"; \
    done; \
    rm -f /root/.git-credentials.rw; \
    git config --system credential."http://gitlab.alibaba-inc.com".helper env; \
    chown -R 1000:1000 /opt/code; \
    echo "cloned code layer:"; ls -1 /opt/code

USER 1000
RUN aliyun plugin install --names aliyun-cli-pai-dsw && aliyun plugin list
