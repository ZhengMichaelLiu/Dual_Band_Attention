def solve_cyclic_patterns():
    N = 16
    OUTPUT_FILE = "cyclic_best_patterns.txt"

    # We will search these K values
    K_VALUES = [2]

    # Store unique valid patterns: key=score, value=list of patterns
    # We use a set for patterns to avoid duplicates (e.g. shift 0 vs shift 8)
    best_results = {}

    print(f"Searching Cyclic Shift patterns for N={N}...\n")

    # 1. Iterate over K (Number of ones)
    for k in K_VALUES:

        # 2. Iterate over Stride (Distance between ones in a row)
        # If k=1, stride is irrelevant (conceptually 0 or N), we just set loop to [0]
        possible_strides = range(1, N) if k > 1 else [0]

        for stride in possible_strides:

            # Max valid start index for a row to fit inside [0, 7]
            # e.g. K=2, Stride=1. Length is 2. Max start is 6 (so [6,7]).
            max_row_start = N - 1 - (k - 1) * stride
            if max_row_start < 0:
                continue # This stride is too big for this K

            # 3. Iterate over Shift (How much the start moves each row)
            # Shift can be negative (moving left) or positive (moving right)
            # Modulo 8 covers both (e.g., -1 is 7).
            for shift in range(N):

                # 4. Iterate over Start Index of Row 0
                for start_0 in range(max_row_start + 1):

                    # --- Generate Grid ---
                    grid_rows = []
                    valid_pattern = True

                    current_start = start_0

                    for row_idx in range(N):
                        # Check if this row fits
                        if current_start > max_row_start:
                            valid_pattern = False
                            break

                        # Generate row cols
                        cols = tuple(current_start + i * stride for i in range(k))
                        grid_rows.append(cols)

                        # Apply Cyclic Shift for next row
                        current_start = (current_start + shift) % N

                    if not valid_pattern:
                        continue

                    # --- Check Column Constraints ---
                    # Each column must have exactly K ones?
                    # Actually, for K=1, each col must have 1. For K=2, each col must have 2.
                    col_counts = [0] * N
                    for cols in grid_rows:
                        for c in cols:
                            col_counts[c] += 1

                    if any(c != k for c in col_counts):
                        continue

                    # --- Score ---
                    score = get_diagonal_score(grid_rows)

                    # --- Save ---
                    if score not in best_results:
                        best_results[score] = []

                    # Add metadata
                    pattern_data = {
                        "k": k,
                        "stride": stride,
                        "shift": shift,
                        "rows": grid_rows
                    }

                    # Avoid duplicates
                    if pattern_data not in best_results[score]:
                        best_results[score].append(pattern_data)

    # --- Output Best Results ---
    sorted_scores = sorted(best_results.keys(), reverse=True)

    if not sorted_scores:
        print("No valid cyclic patterns found.")
        return

    best_score = sorted_scores[0]
    print(f"Search Complete. Best Score: {best_score}/{2*N-1}")

    top_patterns = best_results[best_score][:100] # Limit to 100

    with open(OUTPUT_FILE, "w") as f:
        f.write(f"CYCLIC PATTERN SEARCH (N={N})\n")
        f.write(f"BEST SCORE: {best_score}\n")
        f.write("="*40 + "\n")

        for i, p in enumerate(top_patterns):
            f.write(f"\nPattern #{i+1} (K={p['k']}, Stride={p['stride']}, Shift={p['shift']}):\n")
            write_pattern(f, p['rows'], N)

    print(f"Saved {len(top_patterns)} top patterns to {OUTPUT_FILE}")

def get_diagonal_score(rows):
    covered = set()
    for r, cols in enumerate(rows):
        for c in cols:
            covered.add(r - c)
    return len(covered)

def write_pattern(f_handle, rows, N):
    f_handle.write("   " + " ".join(str(i) for i in range(N)) + "\n")
    f_handle.write("  +" + "-" * (N*2) + "\n")
    for r, cols in enumerate(rows):
        line = ["."] * N
        for c in cols: line[c] = "1"
        f_handle.write(f"{r} | {' '.join(line)}\n")

if __name__ == "__main__":
    solve_cyclic_patterns()