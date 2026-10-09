import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

CSV_PATH = "scripts/evaluation_scores_master.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    
    human_col = "Human_Grade"
    bert_col = "BERTScore"
    
    if human_col not in df.columns or bert_col not in df.columns:
        print(f"Σφάλμα: Δεν βρέθηκαν οι στήλες {human_col} ή/και {bert_col}")
        print("Υπάρχουσες στήλες:", df.columns.tolist())
        return

    valid_df = df.dropna(subset=[human_col, bert_col])
    
    if valid_df.empty:
        print("Δεν βρέθηκαν κοινές εγγραφές με βαθμολογίες.")
        return

    human_scores = valid_df[human_col]
    bert_scores = valid_df[bert_col]

    # Υπολογισμοί
    spearman_r, _ = spearmanr(human_scores, bert_scores)
    pearson_r, _ = pearsonr(human_scores, bert_scores)
    mae = np.mean(np.abs(human_scores - bert_scores))
    mean_human = human_scores.mean()
    mean_bert = bert_scores.mean()

    # Εκτύπωση αποτελεσμάτων έτοιμα για τον πίνακα
    print("=== BERTScore vs Human Grade ===")
    print(f"Sample Count: {len(valid_df)}")
    print(f"Spearman r:   {spearman_r:.4f}")
    print(f"Pearson r:    {pearson_r:.4f}")
    print(f"MAE:          {mae:.4f}")
    print(f"Mean Human:   {mean_human:.4f}")
    print(f"Mean BERT:    {mean_bert:.4f}")

if __name__ == "__main__":
    main()