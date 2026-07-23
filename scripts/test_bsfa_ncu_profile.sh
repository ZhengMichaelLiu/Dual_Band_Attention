#!/bin/bash
set -e

# 1. Get current path (Dual_Band_Attention/scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Get project root directory (the parent directory of the two main folders)
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 3. Define test file path
BSFA_DIR="$ROOT_DIR/Block_Sparse_Flash_Attention"
TEST_TARGET="tests/profile_ncu_kernels.py"
OUT_DIR="$BSFA_DIR/tests/ncu_profile_result"
mkdir -p "$OUT_DIR"

# Metrics covering: coalesced global memory access, warp divergence,
# register pressure, occupancy, global memory throughput.
METRICS="l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
smsp__sass_average_branch_targets_threads_uniform.pct,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
launch__registers_per_thread,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,\
launch__occupancy_limit_blocks,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum.per_second,\
lts__throughput.avg.pct_of_peak_sustained_elapsed"

cd "$BSFA_DIR"

# Only match the actual attention CUDA kernel, filtering out the small ATen
# ops (sort/scan/arange/etc.) used by convert_mask() preprocessing so those
# don't pollute the report/CSV with irrelevant kernel rows.
KERNEL_FILTER="regex:flash_fwd_kernel"

for MODE in base dba; do
    echo "=== Full ncu report (--set full) for mode=$MODE ==="
    ncu --set full --kernel-name "$KERNEL_FILTER" --profile-from-start off --target-processes all \
        -f -o "$OUT_DIR/ncu_report_${MODE}" \
        python "$TEST_TARGET" "$MODE"

    echo "=== Targeted metrics CSV for mode=$MODE ==="
    ncu --metrics "$METRICS" --kernel-name "$KERNEL_FILTER" --profile-from-start off --target-processes all \
        --csv --page raw \
        python "$TEST_TARGET" "$MODE" > "$OUT_DIR/ncu_metrics_${MODE}.csv"
done

echo "Done. Reports in $OUT_DIR"
echo "Open *.ncu-rep with 'ncu-ui <file>.ncu-rep' (GUI) or 'ncu --import <file>.ncu-rep' (CLI text report)."
