#!/bin/bash
# Build llama.cpp as a UGOS-compatible SYCL runtime for `ugos-llm.py runtime`.
#
# Run INSIDE intel/oneapi-basekit:2025.3.2-0-devel-ubuntu22.04 (glibc 2.35):
#   docker run --rm -v /volume1/docker/ugos-llm-build:/out \
#     intel/oneapi-basekit:2025.3.2-0-devel-ubuntu22.04 bash /out/build-runtime.sh
#
# Verified on an iDX6011 (UGOS Pro, glibc 2.36, July 2026). Non-negotiables,
# each learned the hard way — see docs/known-bugs.md section 6:
#   * Ubuntu 22.04 base: anything newer links GLIBC >= 2.38 symbols.
#   * -DGGML_SYCL_DNN=OFF: with oneDNN the build either crashes on
#     glibc-compatible level-zero drivers or re-JITs every prompt batch on
#     the OpenCL path (~2 t/s prompt processing). GEMM goes through oneMKL.
#   * UR adapters, umf and the MKL libraries are dlopen'd/unresolved at
#     build time and DO NOT appear in the ldd closure — they are copied
#     explicitly, and any remaining unresolved dependency fails the build.
#   * The runtime runs on UGREEN's own OpenCL userspace (vendor bundle);
#     no GPU driver is bundled here.
set -euo pipefail

# Two pins are verified end-to-end on an iDX6011; the default builds the
# newer one (b10621, 2026-08-25 — ~18 % faster generation on the qwen3_5
# arch, MTP autodetect/memory-fit, GDN writeback fusion). For a bit-exact
# repro of the July verification (the build Gemma 4's acceptance run and
# MTP numbers were measured on):
#   docker run --rm -e UGOS_LLM_COMMIT=88b47a755c72fed4b22fba0fd262e2d7b7d01583 \
#     -e UGOS_LLM_BUILD=b10143 -v /volume1/docker/ugos-llm-build:/out ...
# If you pick your own pin: avoid b10552-b10575 (broken Q2_K ESIMD window,
# re-landed in b10576) — see docs/known-bugs.md section 6.
COMMIT=${UGOS_LLM_COMMIT:-c1d0e7a004015f23bc0233470b747b596f29b264}   # = b10621, verified Aug 2026
BUILD=${UGOS_LLM_BUILD:-b10621}
OUT=/out/ugos-llm-runtime-$BUILD

echo "=== [1/6] deps ==="
apt-get update -qq
apt-get install -y -qq git wget ca-certificates cmake build-essential curl > /dev/null
LZ_VER=1.28.2
cd /tmp
if wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LZ_VER}/level-zero_${LZ_VER}%2Bu22.04_amd64.deb" -O lz.deb && \
   wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LZ_VER}/level-zero-devel_${LZ_VER}%2Bu22.04_amd64.deb" -O lz-dev.deb; then
  apt-get -o Dpkg::Options::="--force-overwrite" install -y ./lz.deb ./lz-dev.deb > /dev/null
else
  apt-get install -y -qq libze-dev libze1 > /dev/null
fi

echo "=== [2/6] source ==="
git clone --quiet https://github.com/ggml-org/llama.cpp /src
cd /src
git checkout --quiet $COMMIT

echo "=== [3/6] configure + build ==="
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
export SYCL_PROGRAM_COMPILE_OPTIONS="-cl-fp32-correctly-rounded-divide-sqrt"
cmake -B build \
  -DGGML_NATIVE=OFF \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DNN=OFF \
  -DLLAMA_USE_PREBUILT_UI=ON
cmake --build build --config Release -j"$(nproc)"

echo "=== [4/6] package: binaries, build libs, dlopen'd runtime libs ==="
rm -rf "$OUT"          # stale libraries from a previous run must not survive
mkdir -p "$OUT"
cp -P build/bin/llama-server "$OUT/"
find build -name "*.so*" -exec cp -P {} "$OUT/" \;
CMPLR=$(ls -d /opt/intel/oneapi/compiler/2025.*/lib | head -1)
MKL=$(ls -d /opt/intel/oneapi/mkl/2025.*/lib | head -1)
UMF=$(ls -d /opt/intel/oneapi/umf/*/lib | head -1)
cp -P "$CMPLR"/libur_adapter_level_zero.so* "$CMPLR"/libur_adapter_opencl.so* "$OUT/"
cp -P "$UMF"/libumf.so* "$OUT/"
cp -P "$MKL"/libmkl_sycl_blas.so* "$MKL"/libmkl_intel_ilp64.so* \
      "$MKL"/libmkl_tbb_thread.so* "$MKL"/libmkl_core.so* "$OUT/"

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
# `grep || true`: zero matches is the GOOD case, and under pipefail a bare
# grep exit 1 would abort the whole script right here.
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

echo "$COMMIT $BUILD ubuntu22.04 oneapi-2025.3.2 GGML_SYCL_DNN=OFF" > "$OUT/BUILD_INFO"
cd /out && tar czf "ugos-llm-runtime-$BUILD.tar.gz" "ugos-llm-runtime-$BUILD"
echo "BUILD OK: $OUT ($(du -sh "$OUT" | cut -f1))"
