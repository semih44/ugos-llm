#!/bin/bash
# Build llama.cpp as a UGOS-compatible VULKAN runtime (experiment vs SYCL).
#
# VERDICT (Aug 2026, Meteor Lake iGPU): built, deployed, TESTED, REJECTED —
# qwen35moe hangs on first inference (ANV/Xe-LPG MoE shader bug, coopmat
# on or off), and the dense 9B runs ~45 % slower than on the SYCL runtime
# (6.7 vs ~13 t/s). Kept because it reproduces the whole experiment in one
# container run — retry when mesa or llama.cpp's Vulkan MoE path moves.
# Full story incl. the deployment wrapper env: docs/known-bugs.md section 7.
# Run INSIDE ubuntu:22.04 (glibc 2.35):
#   docker run --rm -v /volume1/docker/ugos-llm-build:/out \
#     ubuntu:22.04 bash /out/build-runtime-vulkan.sh
#
# Differences vs the SYCL recipe (build-runtime.sh):
#   * Vulkan/ANV needs NO vendor OpenCL userspace and NO level-zero —
#     mesa's libvulkan_intel.so talks straight to the kernel DRM (6.12.30
#     on the NAS is recent enough for Meteor Lake).
#   * mesa comes from the kisak PPA (current mesa built for jammy/glibc
#     2.35 — stock jammy mesa predates Meteor Lake ANV support).
#   * glslc (shader compiler, build-time only) comes from the LunarG repo.
#   * The bundled ICD manifest uses a bare library_path so the loader
#     resolves the driver via LD_LIBRARY_PATH inside the runtime dir.
#   * -j4 on purpose: a 16 GB model is resident on the NAS during this
#     build; full -j would swap the box.
set -euo pipefail

COMMIT=c1d0e7a004015f23bc0233470b747b596f29b264   # = b10621, same as SYCL build
BUILD=b10621-vulkan
OUT=/out/ugos-llm-runtime-$BUILD

echo "=== [1/6] deps (kisak mesa + LunarG glslc) ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git wget curl ca-certificates cmake build-essential \
  gnupg software-properties-common pkg-config libcurl4-openssl-dev > /dev/null
add-apt-repository -y ppa:kisak/kisak-mesa > /dev/null
wget -qO /etc/apt/trusted.gpg.d/lunarg.asc \
  https://packages.lunarg.com/lunarg-signing-key-pub.asc
wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
  https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
apt-get update -qq
apt-get install -y -qq mesa-vulkan-drivers libvulkan1 libvulkan-dev > /dev/null
apt-get install -y -qq vulkan-sdk > /dev/null || \
  apt-get install -y -qq shaderc glslc > /dev/null || true
command -v glslc > /dev/null || { echo "FAIL: no glslc available"; exit 1; }
echo "mesa: $(dpkg -s mesa-vulkan-drivers | awk '/^Version/{print $2}')"

echo "=== [2/6] source ==="
git clone --quiet https://github.com/ggml-org/llama.cpp /src
cd /src
git checkout --quiet $COMMIT

echo "=== [3/6] configure + build (Vulkan) ==="
cmake -B build \
  -DGGML_NATIVE=OFF \
  -DGGML_VULKAN=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_USE_PREBUILT_UI=ON
cmake --build build --config Release -j4

echo "=== [4/6] package: binaries, build libs, vulkan loader + ANV driver ==="
rm -rf "$OUT"
mkdir -p "$OUT"
cp -P build/bin/llama-server "$OUT/"
find build -name "*.so*" -exec cp -P {} "$OUT/" \;
LIBDIR=/usr/lib/x86_64-linux-gnu
cp -P "$LIBDIR"/libvulkan.so.1* "$OUT/"
cp -L "$LIBDIR"/libvulkan_intel.so "$OUT/"
sed 's#"library_path" *: *"[^"]*"#"library_path": "libvulkan_intel.so"#' \
  /usr/share/vulkan/icd.d/intel_icd.x86_64.json > "$OUT/intel_icd.x86_64.json"
grep -q 'libvulkan_intel' "$OUT/intel_icd.x86_64.json" || { echo "FAIL: ICD rewrite"; exit 1; }
# libbsd -> libmd is resolved too late for the 3-pass closure below, and the
# final ldd gate can't see it (the build container has libmd system-wide,
# the target host may not). Learned on the first deploy: bundle explicitly.
cp -L "$LIBDIR"/libmd.so.0* "$OUT/" 2>/dev/null || true

echo "=== [5/6] ldd closure — unresolved is a hard error ==="
GLIBC_RE='^(libc|libm|libdl|libpthread|librt|libresolv|libnsl|libutil|ld-linux|libanl|libnss)'
for pass in 1 2 3; do
  for f in "$OUT"/llama-server "$OUT"/*.so*; do
    [ -f "$f" ] || continue
    ldd "$f" 2>/dev/null | awk '/=>/ {print $1, $3}' | while read -r name path; do
      base=$(basename "$name")
      [[ "$base" =~ $GLIBC_RE ]] && continue
      [ -z "$path" ] || [ "$path" = "not" ] && continue
      if [ ! -e "$OUT/$base" ]; then
        cp -L "$path" "$OUT/$base"
        echo "  + $base"
      fi
    done
  done
done
UNRESOLVED=$(cd "$OUT" && for f in llama-server *.so*; do
  LD_LIBRARY_PATH="$OUT" ldd "$f" 2>/dev/null | grep "not found" || true
done | sort -u)
if [ -n "$UNRESOLVED" ]; then
  echo "FAIL: unresolved dependencies remain:"; echo "$UNRESOLVED"; exit 1
fi
echo "closure complete"

echo "=== [6/6] ELF gate: no GLIBC > 2.36 allowed ==="
FAIL=0
for f in "$OUT"/*; do
  bad=$(objdump -T "$f" 2>/dev/null | grep -oE 'GLIBC_2\.[0-9]+' | sort -uV | awk -F. '$2 > 36' || true)
  if [ -n "$bad" ]; then echo "FAIL: $f requires $bad"; FAIL=1; fi
done
[ "$FAIL" = "1" ] && { echo "ELF GATE FAILED"; exit 1; }
echo "ELF gate passed"

echo "$COMMIT $BUILD ubuntu22.04 kisak-mesa GGML_VULKAN=ON" > "$OUT/BUILD_INFO"
cd /out && tar czf "ugos-llm-runtime-$BUILD.tar.gz" "ugos-llm-runtime-$BUILD"
echo "BUILD OK: $OUT ($(du -sh "$OUT" | cut -f1))"
