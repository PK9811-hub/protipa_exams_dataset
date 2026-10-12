from datasets import load_dataset
import pandas as pd

def main():
    private_repo = "ilsp/greek-protipa-exams-private"
    
    print(f"Downloading the full dataset from: {private_repo}")
    dataset = load_dataset(private_repo, split="test")
    
    print("Filtering: Removing 2019 exams...")
    public_dataset = dataset.filter(lambda x: x['year'] != '2019')
    
    print("Converting to Pandas DataFrame...")
    df = public_dataset.to_pandas()
    
    def extract_filename(img_data):
        if img_data is None:
            return None
            
        if isinstance(img_data, list):
            paths = []
            for item in img_data:
                if isinstance(item, dict) and 'path' in item:
                    paths.append(item['path'])
                elif isinstance(item, str):
                    paths.append(item)
            return ", ".join(paths) if paths else None

        if isinstance(img_data, dict) and 'path' in img_data:
            return img_data['path']
            
        if isinstance(img_data, str):
            return img_data
            
        return "image_reference.png"

    col_name = 'images'
    if col_name in df.columns:
        print(f"Cleaning column '{col_name}' from bytes...")
        df[col_name] = df[col_name].apply(extract_filename)
    else:
        print(f"Warning: Column '{col_name}' not found in the dataset.")
    
    output_filename = "protipa_exams_public.xlsx"
    print(f"Saving to {output_filename}...")
    
    df.to_excel(output_filename, index=False)
    
    print("Completed successfully!")

if __name__ == "__main__":
    main()