#!/usr/bin/env bash

set -Eeuo pipefail

cd /app

exec "$@"
