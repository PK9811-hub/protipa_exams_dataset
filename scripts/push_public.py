from datasets import load_dataset
import os

def main():
    private_repo = "ilsp/greek-protipa-exams-private"
    public_repo = "ilsp/greek-protipa-exams"
    
    print(f"Κατέβασμα του πλήρους dataset από: {private_repo}")
    dataset = load_dataset(private_repo, split="test")
    
    print("Φιλτράρισμα: Αφαίρεση θεμάτων του 2026...")
    public_dataset = dataset.filter(lambda x: x['year'] != '2026')
    
    print(f"Ανέβασμα του φιλτραρισμένου dataset στο: {public_repo}")
    public_dataset.push_to_hub(public_repo, private=False)
    
    print("Ολοκληρώθηκε με επιτυχία!")

if __name__ == "__main__":
    main()

#Τρέχουμε uv run scripts/push_public.py στο τερματικό