#!/usr/bin/env bash
set -Eeuo pipefail
# Already completed: SFT, GBPO.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" Sports GRPO GSPO DAPO GPPO DUAL_PPO
