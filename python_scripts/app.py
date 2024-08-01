import sys
import pandas as pd

def main():
    try:
        file_path = sys.argv[1]
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("error|Unsupported file type")
            return
        
        rows, columns = df.shape
        print(f"success|{rows}|{columns}")
    except Exception as e:
        print(f"error|{str(e)}")

if __name__ == "__main__":
    main()
