import json
import pandas as pd

def format_longbench_result(json_path):
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create a dictionary to store the results
    results = {}

    # Process each entry
    for key, value in data.items():
        if 'repobench-p' in key:
            key = key.replace('repobench-p', 'repobench_p')
        # Split the key into dataset name and method
        parts = key.split('-', 1)
        dataset = parts[0]
        method = parts[1].replace('.jsonl', '')

        # Initialize the dict if not exists
        if method not in results:
            results[method] = {}

        # Store the value
        results[method][dataset] = value

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')

    # Sort columns by dataset name
    df = df.sort_index(axis=1)

    # Calculate averages row-wise and add as a column
    df['average'] = df.mean(axis=1)

    # Round all values to 2 decimal places
    df = df.round(2)

    return df

if __name__ == "__main__":
    # Example usage
    result_df = format_longbench_result('./pred/llama3.1-8b-instruct/result.json')
    print("\nFormatted Results:")
    print(result_df.to_string())
    result_df.to_excel('./pred/llama3.1-8b-instruct/result.xlsx')
    print("\nExcel file saved to ./pred/llama3.1-8b-instruct/result.xlsx")