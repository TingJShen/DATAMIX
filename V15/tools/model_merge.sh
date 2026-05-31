#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/run/global_step_xxx [target_dir]" >&2
    exit 2
fi

checkpoint_dir="$1"
target_dir="${2:-${checkpoint_dir}_merged}"
model_path="${checkpoint_dir}/actor"

python -m tools.model_merge merge \
    --backend fsdp \
    --local_dir "${model_path}" \
    --target_dir "${target_dir}"
