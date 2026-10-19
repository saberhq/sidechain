# Pin numpy, OpenBLAS, torch and MKL to their AVX2 code paths on an AVX-512 AMD box (Zen 4: EPYC 9xx4), so it computes
# what the Zen 3 boxes our records were made on compute (EPYC 7763, 7742). Does nothing on any other CPU.
# Source it in the shell that launches a kit's workers, on the box (an /etc/environment entry does not reach ssh commands):
#
#   . ~/code/sidechain/scripts/boxkit/box_cpu_env.sh && echo "$BOX_CPU_ENV"
#
# Measured: /sidechain-brev trap 28 (T118, T125's box night of 2026-10-18).
_box_cpuinfo="${BOX_CPUINFO:-/proc/cpuinfo}"
if grep -q '^vendor_id.*AuthenticAMD' "$_box_cpuinfo" 2>/dev/null && grep -qw avx512f "$_box_cpuinfo" 2>/dev/null; then
  export OPENBLAS_CORETYPE=ZEN
  export NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
  export ATEN_CPU_CAPABILITY=avx2
  export MKL_ENABLE_INSTRUCTIONS=AVX2
  export BOX_CPU_ENV="pinned to AVX2 (AMD with AVX-512)"
else
  export BOX_CPU_ENV="unchanged (no AVX-512 AMD CPU)"
fi
unset _box_cpuinfo
