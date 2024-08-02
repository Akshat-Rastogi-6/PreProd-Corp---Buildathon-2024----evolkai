import pandas as pd
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Data Transformation")
    parser.add_argument('file_path', type=str, help="Path to the CSV file")
    parser.add_argument('--remove_features', type=str, default='', help="Comma-separated list of features to remove")
    parser.add_argument('--label_encoding', type=str, default='', help="Comma-separated list of features to encode")

    args = parser.parse_args()
    file_path = args.file_path
    remove_features = [x.strip() for x in args.remove_features.split(',')] if args.remove_features else []
    label_encoding = [x.strip() for x in args.label_encoding.split(',')] if args.label_encoding else []

    try:
        # Load data
        df = pd.read_csv(file_path)

        # Remove features
        if remove_features:
            df.drop(columns=[f for f in remove_features if f in df.columns], inplace=True)

        # Label encoding
        if label_encoding:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for feature in label_encoding:
                if feature in df.columns:
                    df[feature] = le.fit_transform(df[feature])

        # Prepare output
        output = {
            'rows': df.shape[0],
            'columns': list(df.columns),
            'data': df.head().to_dict(orient='records')
        }

        print(json.dumps(output))
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
