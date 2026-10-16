import sys
import re
from collections import defaultdict
from pathlib import Path
from inspect_ai.log import read_eval_log

def get_model_and_dataset(filename):
    """Εξάγει το μοντέλο και το dataset από το όνομα του .eval αρχείου."""
    fname = filename.lower()
    
    if "krikri" in fname: model = "KriKri-8B"
    elif "llama" in fname: model = "Llama-8B"
    elif "gemma" in fname: model = "Gemma-4-26B"
    elif "qwen" in fname: model = "Qwen3-32B"
    else: model = "Unknown"
    
    if "prot" in fname: dataset = "Prot-Ex"
    else: dataset = "Pan-Ex"
        
    return model, dataset

def clean_subject_name(subj):
    """Μετατρέπει τα ονόματα των μαθημάτων για να ταιριάζουν με το Paper."""
    mapping = {
        "ancient_greek": "Ancient Greek",
        "economics": "Economics",
        "physics": "Physics",
        "history": "History",
        "latin": "Latin",
        "mathematics": "Mathematics",
        "math": "Mathematics",
        "greek_language": "Greek Language",
        "computer_science": "Computer Science",
        "chemistry": "Chemistry",
        "biology": "Biology",
        "religious_studies": "Religious Studies"
    }
    return mapping.get(subj.lower().strip(), subj.replace("_", " ").title())

def aggregate_scores(base_dir):
    print(f"Scanning directory: {base_dir} for ZERO-SHOT eval logs...\n")
    
    # Αποθήκευση βαθμολογιών: scores[dataset][subject][model] = [scores...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    path = Path(base_dir)
    
    eval_files = [f for f in path.rglob("*.eval") if "0-shot" in f.name or "zero-shot" in str(f)]
    
    if not eval_files:
        print("No zero-shot .eval files found in the specified directory.")
        return

    for log_path in eval_files:
        try:
            log = read_eval_log(str(log_path))
        except Exception:
            continue
            
        if not log.samples:
            continue

        model, dataset = get_model_and_dataset(log_path.name)

        for sample in log.samples:
            metadata = sample.metadata or {}
            raw_subject = metadata.get("subject", "Unknown")
            subject = clean_subject_name(raw_subject)
            
            if sample.scores:
                for scorer_name, score_obj in sample.scores.items():
                    # Στοχεύουμε ΑΥΣΤΗΡΑ τον Mistral Judge
                    if scorer_name != "generic_judge_scorer1":
                        continue
                        
                    val = None
                    exp = None
                    if isinstance(score_obj, dict):
                        val = score_obj.get("value")
                        exp = score_obj.get("explanation")
                    else:
                        val = getattr(score_obj, "value", None)
                        exp = getattr(score_obj, "explanation", None)

                    # Regex διάσωση βαθμού αν ο Mistral απάντησε με JSON
                    if val is None and exp:
                        match = re.search(r'"grade"\s*:\s*([\d\.]+)', str(exp))
                        if match:
                            val = float(match.group(1))

                    if val is not None:
                        try:
                            data[dataset][subject][model].append(float(val))
                        except (ValueError, TypeError):
                            pass

    models_order = ["KriKri-8B", "Llama-8B", "Gemma-4-26B", "Qwen3-32B"]
    
    # Εκτύπωση αποτελεσμάτων στη μορφή των LaTeX Tables
    for dataset in ["Pan-Ex", "Prot-Ex"]:
        if dataset not in data:
            continue
            
        print(f"=== {dataset} BENCHMARK (MISTRAL ZERO-SHOT SCORES) ===")
        print(f"| Subject | " + " | ".join(models_order) + " |")
        print("| :--- | " + " | ".join([":---:"] * len(models_order)) + " |")
        
        dataset_totals = {m: [] for m in models_order}
        
        # Γραμμές Μαθημάτων
        for subject in sorted(data[dataset].keys()):
            row_vals = []
            for model in models_order:
                scores = data[dataset][subject][model]
                if scores:
                    avg_pct = (sum(scores) / len(scores)) * 100
                    row_vals.append(f"{avg_pct:.2f}")
                    dataset_totals[model].extend(scores)
                else:
                    row_vals.append("-")
            print(f"| {subject} | " + " | ".join(row_vals) + " |")
            
        # Γραμμή Aggregate
        agg_vals = []
        for model in models_order:
            scores = dataset_totals[model]
            if scores:
                avg_pct = (sum(scores) / len(scores)) * 100
                agg_vals.append(f"{avg_pct:.2f}")
            else:
                agg_vals.append("-")
        print(f"| **Aggregate** | **" + "** | **".join(agg_vals) + "** |\n")

if __name__ == "__main__":
    default_dir = "/home/shared/acl_2027/inspect-ai/"
    target_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    aggregate_scores(target_dir)