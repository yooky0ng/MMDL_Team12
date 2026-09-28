#!/usr/bin/env bash
set -euo pipefail

# 이 서버에서 Triton이 요구하는 libcuda.so 링크를 임시로 생성
cuda_driver_path="$(ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so.1" {print $NF; exit}')"
if [[ -z "${cuda_driver_path}" ]]; then
  echo "Could not locate libcuda.so.1 via ldconfig." >&2
  exit 1
fi

cuda_link_dir="$(mktemp -d -t mmdl-libcuda-XXXXXX)"
cleanup_cuda_link() {
  rm -f "${cuda_link_dir}/libcuda.so"
  rmdir "${cuda_link_dir}" 2>/dev/null || true
}
trap cleanup_cuda_link EXIT
ln -s "${cuda_driver_path}" "${cuda_link_dir}/libcuda.so"
export LIBRARY_PATH="${cuda_link_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"

python scripts/eval_mmmu.py "$@"
