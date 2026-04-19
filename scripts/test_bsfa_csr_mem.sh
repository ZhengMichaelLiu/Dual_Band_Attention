#!/bin/bash

# 1. Get current path (Dual_Band_Attention/scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Get project root directory (the parent directory of the two main folders)
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 3. Define test file path
BSFA_DIR="$ROOT_DIR/Block_Sparse_Flash_Attention"
TEST_TARGET="tests/test_csr_mem.py"

# 4. Run python
cd "$BSFA_DIR" && python "$TEST_TARGET"