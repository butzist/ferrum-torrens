#!/usr/bin/env bash
# Build the CUPTI activity tracer (needs the devShell's CUDA toolkit).
#
# The tracer is preloaded into a CUDA process and writes per-kernel device time
# to $CUPTI_TRACE_OUT.<pid>. It exists because neither packaged profiler works
# here: `ncu` needs admin for performance counters (RmProfilingAdminOnly=1) and
# `nsys`' qdstrm importer aborts against driver 595.112 / CUDA 13.2 (the store's
# nsight-systems is the CUDA 12.9 build). CUPTI *activity* tracing needs neither.
#
#   scripts/cupti-trace/build.sh
#   CUPTI_TRACE_OUT=/tmp/trace LD_PRELOAD=$PWD/scripts/cupti-trace/libcupti_trace.so \
#     prlimit --as=68719476736 <binary> <args>
#   python3 scripts/cupti-trace/decode-steps.py /tmp/trace.<pid>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cupti=${CUPTI_ROOT:-${CUDA_HOME:?set CUDA_HOME (the devShell exports it)}}
gcc -O2 -fPIC -shared \
    -I"$cupti/include" \
    -o "$here/libcupti_trace.so" "$here/cupti_trace.c" \
    -L"$cupti/lib64" -L"$cupti/lib" -lcupti
echo "built $here/libcupti_trace.so"
