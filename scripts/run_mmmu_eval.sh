#!/usr/bin/env bash
set -euo pipefail

# 일부 Linux 환경에서는 Triton이 링크할 libcuda.so가 별도로 필요합니다.
cuda_driver_path=""
if command -v ldconfig >/dev/null 2>&1; then
  cuda_driver_path="$(ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so.1" && !found {print $NF; found = 1}')" || cuda_driver_path=""
fi

if [[ -f "$cuda_driver_path" ]]; then
  cuda_link_dir="$(mktemp -d -t mmdl-libcuda-XXXXXX)"
  cleanup_cuda_link() {
    rm -f "${cuda_link_dir}/libcuda.so"
    rmdir "${cuda_link_dir}" 2>/dev/null || true
  }
  trap cleanup_cuda_link EXIT
  ln -s "$cuda_driver_path" "${cuda_link_dir}/libcuda.so"
  export LIBRARY_PATH="${cuda_link_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

python scripts/eval_mmmu.py "$@"
