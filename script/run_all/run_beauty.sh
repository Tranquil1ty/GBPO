#!/usr/bin/env bash
set -Eeuo pipefail
# Already completed: SFT, GBPO, GSPO.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" Beauty GRPO DAPO GPPO DUAL_PPO
