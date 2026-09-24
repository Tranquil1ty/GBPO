#!/usr/bin/env bash
set -Eeuo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" Toys SFT GBPO GRPO GSPO DAPO GPPO DUAL_PPO
