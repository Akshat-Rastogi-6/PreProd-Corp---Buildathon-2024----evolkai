import sys
import pandas as pd
import json

def analyze_file(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get the number of rows and columns
        num_rows, num_columns = df.shape
        
        # Create a dictionary with the results
        result = {
            'rows': num_rows,
            'columns': num_columns
        }
        
        # Print the result as JSON
        print(json.dumps(result))
    except Exception as e:
        # Print the error as JSON
        print(json.dumps({'error': str(e)}), file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        analyze_file(file_path)
    else:
        print(json.dumps({'error': 'No file path provided'}), file=sys.stderr)
