from datasets import load_dataset
import pandas as pd

def main():
    private_repo = "ilsp/greek-protipa-exams-private"
    
    print(f"Κατέβασμα του πλήρους dataset από: {private_repo}")
    dataset = load_dataset(private_repo, split="test")
    
    print("Φιλτράρισμα: Αφαίρεση θεμάτων του 2019...")
    public_dataset = dataset.filter(lambda x: x['year'] != '2019')
    
    print("Μετατροπή σε Pandas DataFrame...")
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
        print(f"Καθαρισμός της στήλης '{col_name}' από τα bytes...")
        df[col_name] = df[col_name].apply(extract_filename)
    else:
        print(f"Προσοχή: Η στήλη '{col_name}' δεν βρέθηκε στο dataset.")
    
    output_filename = "protipa_exams_public.xlsx"
    print(f"Αποθήκευση σε {output_filename}...")
    
    df.to_excel(output_filename, index=False)
    
    print("Ολοκληρώθηκε με επιτυχία!")

if __name__ == "__main__":
    main()