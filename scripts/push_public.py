from datasets import load_dataset
import os

def main():
    private_repo = "ilsp/greek-protipa-exams-private"
    public_repo = "ilsp/greek-protipa-exams"
    
    print(f"Downloading the full dataset from: {private_repo}")
    dataset = load_dataset(private_repo, split="test")
    
    print("Filtering: Removing 2019 exams...")
    public_dataset = dataset.filter(lambda x: x['year'] != '2019')
    
    print(f"Pushing the filtered dataset to: {public_repo}")
    public_dataset.push_to_hub(public_repo, private=False)
    
    print("Completed successfully!")

if __name__ == "__main__":
    main()

# Run using: uv run scripts/push_public.py in the terminal