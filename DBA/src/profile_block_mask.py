import torch
import triton
import pandas as pd
import gc
from DBA.src.dual_band_attention import (
    create_block_mask_pytorch,
    create_block_mask_triton_only_softmax,
    create_block_mask_softmax_gemm
)

def benchmark():
    B = 1
    H = 32
    D = 128
    block_size = 128

    seq_lens = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    patch_sizes = [8, 16, 32]
    causals = [True, False]

    methods = {
        'PyTorch': create_block_mask_pytorch,
        'Triton_Softmax': create_block_mask_triton_only_softmax,
        'Softmax_GEMM': create_block_mask_softmax_gemm
    }

    results = []

    for seq_len in seq_lens:
        for patch_size in patch_sizes:
            for causal in causals:
                print(f"Benchmarking L={seq_len}, P={patch_size}, causal={causal}")

                try:
                    q = torch.randn(B, H, seq_len, D, device='cuda', dtype=torch.float16)
                    k = torch.randn(B, H, seq_len, D, device='cuda', dtype=torch.float16)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM allocating inputs for L={seq_len}. Skipping.")
                    torch.cuda.empty_cache()
                    continue

                for name, func in methods.items():
                    try:
                        # Warmup
                        _ = func(q, k, block_size=block_size, patch_size=patch_size, causal=causal)
                        torch.cuda.synchronize()

                        # Benchmark using triton.testing.do_bench
                        ms = triton.testing.do_bench(
                            lambda: func(q, k, block_size=block_size, patch_size=patch_size, causal=causal),
                            quantiles=None
                        )

                        results.append({
                            'Seq Len': seq_len,
                            'Patch Size': patch_size,
                            'Causal': causal,
                            'Method': name,
                            'Time (ms)': ms
                        })
                    except torch.cuda.OutOfMemoryError:
                        results.append({
                            'Seq Len': seq_len,
                            'Patch Size': patch_size,
                            'Causal': causal,
                            'Method': name,
                            'Time (ms)': float('inf')
                        })
                    except Exception as e:
                        print(f"Error with {name} for L={seq_len}: {e}")
                        results.append({
                            'Seq Len': seq_len,
                            'Patch Size': patch_size,
                            'Causal': causal,
                            'Method': name,
                            'Time (ms)': None
                        })
                    finally:
                        torch.cuda.empty_cache()
                        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv('block_mask_benchmark_results.csv', index=False)
    print(df.to_string())

if __name__ == '__main__':
    benchmark()
