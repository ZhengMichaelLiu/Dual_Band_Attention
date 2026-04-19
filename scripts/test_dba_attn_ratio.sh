#!/bin/bash

# 1. Get current path (Dual_Band_Attention/scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" && pwd )"

# 2. Get project root directory (the parent directory of the two main folders)
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 3. Define test file path
cd "$ROOT_DIR"

# 4. Run python
echo "Starting DBA Attention Ratio Test..."
python Dual_Band_Attention/eval/Speed/whole_model/test_dba_attn_ratio.py
