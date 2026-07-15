# syntax=docker/dockerfile:1
# =============================================================================
# pai-rec sandbox template. Adds a release-pinned source snapshot at /opt/code
# and the PAI recommendation plugins to the shared base.
#
#   sandbox/build.sh pairec
#
# Register in AgentRun as a code-interpreter template with network mode PUBLIC,
# under the same account_id + region the agent service uses, and point
# settings.templates.pairec.name at it.
# =============================================================================

FROM sandbox-base:latest

# Release-pinned snapshot of the source repositories (one per subdirectory) the
# agent greps and reads when the knowledge base can't answer. Baked rather than
# NAS-mounted so grep/read hit local disk instead of NFS. Bump the URL to ship
# newer source; pass an empty value to build without the layer.
ARG CODE_ARCHIVE_URL=https://pai-rag.oss-cn-hangzhou.aliyuncs.com/production_artifacts/pairec_code/pairec_code_20260623.tar.gz

# Root-owned + world-readable: this template's /opt/code is a read-only snapshot
# (templates.pairec has no code_writable), so the runtime user only ever reads it.
# --strip-components=1 drops the archive's single top-level wrapper dir so repos
# land at /opt/code/<repo> rather than /opt/code/<wrapper>/<repo>.
USER root
RUN if [ -n "${CODE_ARCHIVE_URL}" ]; then \
      curl -fsSL "${CODE_ARCHIVE_URL}" -o /tmp/code.tar.gz \
      && tar -xzf /tmp/code.tar.gz --strip-components=1 -C /opt/code \
      && rm -f /tmp/code.tar.gz \
      && chmod -R a+rX /opt/code \
      && echo "baked code layer:" && ls -1 /opt/code ; \
    fi

# Installed AS uid 1000 so they land in the agent's $HOME (~/.aliyun) and are
# available to skill scripts at run time. Requires network access during build.
USER 1000
RUN aliyun plugin install --names \
       aliyun-cli-eas \
       aliyun-cli-pairecservice \
       aliyun-cli-pai-dsw \
       aliyun-cli-paifeaturestore \
 && aliyun plugin list
