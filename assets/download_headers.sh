#!/usr/bin/env bash
set -eu

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
version_file="$root_dir/ONNXRUNTIME_VERSION"
version="$(tr -d '[:space:]' < "$version_file")"

url="https://raw.githubusercontent.com/microsoft/onnxruntime/${version}/include/onnxruntime/core/session/onnxruntime_c_api.h"
out="$root_dir/assets/onnxruntime_c_api.h"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$url" -o "$out"
else
  wget -q -O "$out" "$url"
fi

echo "updated: $out ($version)"
