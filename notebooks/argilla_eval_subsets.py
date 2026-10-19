import os
import ast
import base64
import pandas as pd
import argilla as rg
from io import BytesIO
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
API_URL = os.getenv("ARGILLA_API_URL")
API_KEY = os.getenv("ARGILLA_API_KEY")
WORKSPACE_NAME = "pass-or-fail"
HF_REPO = "ilsp/panellinies-exams-dataset"

EXCEL_PATH = "aggregate_eval_scores.xlsx"
SHEETS = {
    "evaluation_scores_protipa": "pass-or-fail-prot-eval",
    "evaluation_scores_panellinies": "pass-or-fail-pan-eval"
}

def extract_clean_text(val, default_key):
    if pd.isna(val): return "N/A"
    val_str = str(val).strip()
    if val_str.startswith("{") and val_str.endswith("}"):
        try:
            parsed_dict = ast.literal_eval(val_str)
            if isinstance(parsed_dict, dict):
                return str(parsed_dict.get(default_key, list(parsed_dict.values())[0])).strip()
        except (ValueError, SyntaxError):
            pass
    return val_str

def clean_val(val):
    return "N/A" if pd.isna(val) else str(val).strip()

def images_to_markdown(images_list):
    """Μετατρέπει τις εικόνες του HF σε Base64 Markdown για το Argilla"""
    if not images_list: return ""
    md_elements = []
    for idx, img in enumerate(images_list):
        if img is None: continue
        try:
            if hasattr(img, "save"):
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                md_elements.append(f'<img src="data:image/png;base64,{b64_str}" alt="Εικόνα {idx+1}" style="max-width: 100%; height: auto;" />')
            elif isinstance(img, str) and img.strip():
                md_elements.append(f"![Εικόνα {idx+1}]({img})")
        except Exception:
            pass
    return "\n\n".join(md_elements)


def create_or_get_argilla_dataset(client, dataset_name, workspace_name):
    existing = client.datasets(name=dataset_name, workspace=workspace_name)
    if existing:
        print(f"Το dataset '{dataset_name}' υπάρχει ήδη. Προσθήκη/Ανανέωση εγγραφών...")
        return existing
        
    print(f"Δημιουργία νέου Dataset: {dataset_name}...")
    
    fields = [
        rg.TextField(name="dataset", title="Dataset"),
        rg.TextField(name="subject", title="Subject"),
        rg.TextField(name="model_evaluated_field", title="Evaluated Model"), 
        rg.TextField(name="question_id", title="Question ID"),
        rg.TextField(name="input_question", title="Input Question & Images", use_markdown=True),
        rg.TextField(name="reference_target", title="Reference Target", use_markdown=True),
        rg.TextField(name="model_answer", title="Model Answer", use_markdown=True),
        rg.TextField(name="human_grade", title="Human Grade"),
        rg.TextField(name="human_explanation", title="Human Explanation", use_markdown=True),
        rg.TextField(name="gemma_score", title="Gemma Judge Score"),
        rg.TextField(name="gemma_explanation", title="Gemma Explanation", use_markdown=True),
        rg.TextField(name="mistral_score", title="Mistral Judge Score"),
        rg.TextField(name="mistral_explanation", title="Mistral Explanation", use_markdown=True),
        rg.TextField(name="bert_score", title="BERTScore")
    ]
    
    questions = [
        rg.RatingQuestion(
            name="ref_completeness",
            title="1. [Reference] Πληρότητα Πρότυπης Απάντησης",
            values=[1, 2, 3, 4, 5],
            description="1 = Απλή οδηγία/rubric διόρθωσης, 5 = Πλήρης και άρτια λύση",
            required=True
        ),
        rg.LabelQuestion(
            name="student_naturalness",
            title="2. [Μαθητής] Η απάντηση έχει φυσική ροή λόγου (απαντάει σαν πραγματικός μαθητής);",
            labels=["Ναι", "Όχι"],
            description="Ναι = Φυσική ροή, Όχι = Ρομποτικό/Μηχανικό ύφος",
            required=True
        ),
        rg.LabelQuestion(
            name="student_loop",
            title="3. [Μαθητής] Το μοντέλο-μαθητής πέφτει σε συνεχή λούπα (repetitive/hallucinated);",
            labels=["Ναι", "Όχι"],
            required=True
        ),
        rg.LabelQuestion(
            name="best_alignment",
            title="4. [Κριτές] Ποιος κριτής προσέγγισε καλύτερα την ανθρώπινη λογική (human explanation);",
            labels=["Μόνο το Gemma", "Μόνο το Mistral", "Και οι δύο εξίσου", "Κανένας", "N/A"],
            required=True
        ),
        rg.RatingQuestion(
            name="gemma_quality",
            title="5. [Κριτές] Ποιότητα εξήγησης Gemma",
            values=[1, 2, 3, 4, 5],
            description="1=Λάθος λογική/Παραίσθηση, 5=Άψογη τεκμηρίωση",
            required=True
        ),
        rg.RatingQuestion(
            name="mistral_quality",
            title="6. [Κριτές] Ποιότητα εξήγησης Mistral",
            values=[1, 2, 3, 4, 5],
            description="1=Λάθος λογική/Παραίσθηση, 5=Άψογη τεκμηρίωση",
            required=True
        ),
        rg.TextQuestion(
            name="notes",
            title="7. Σχόλια (Προαιρετικό)",
            description="Σημειώσεις για την αξιολόγηση",
            required=False
        )
    ]

    guidelines = """**Οδηγίες Ποιοτικού Ελέγχου LLM-as-a-Judge & Μοντέλου-Μαθητή:**
1. **[Reference] Πληρότητα:** Ελέγξτε αν το Reference Target είναι μια πλήρης απάντηση (5) ή απλώς οδηγίες διόρθωσης/βήματα (1).
2. **[Μαθητής] Φυσικότητα & Λούπες:** Διαβάστε την απάντηση του μοντέλου-μαθητή. Απαντάει με φυσική ροή σαν μαθητής (ακόμα και αν κάνει λάθος) ή ρομποτικά; Μήπως "κόλλησε" σε επαναλαμβανόμενη λούπα;
3. **[Κριτές] Σύγκριση με τον Άνθρωπο:** Δείτε τον βαθμό και την εξήγηση του ανθρώπου έναντι των Gemma και Mistral.
4. **[Κριτές] Ποιότητα Εξηγήσεων:** Αξιολογήστε την τεκμηρίωση των LLM-κριτών. Προσοχή στην "τυφλή επιείκεια" και την "υπερβολική αυστηρότητα". Τιμωρήστε εξηγήσεις που εφευρίσκουν ανύπαρκτα λάθη (hallucinations)."""

    metadata_properties = [
        rg.TermsMetadataProperty(name="model_evaluated_meta", title="Evaluated Model"),
        rg.TermsMetadataProperty(name="subject_meta", title="Subject")
    ]

    distribution = rg.TaskDistribution(min_submitted=3)
    
    settings = rg.Settings(
        fields=fields, 
        questions=questions, 
        guidelines=guidelines, 
        distribution=distribution,
        metadata=metadata_properties
    )
    
    dataset = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings)
    dataset.create()
    return dataset


def main():
    if not API_URL or not API_KEY:
        raise ValueError("Παρακαλώ ορίστε τα ARGILLA_API_URL και ARGILLA_API_KEY στο .env")

    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    
    try:
        workspace = client.workspaces(name=WORKSPACE_NAME)
        if workspace:
            for user in ["pkyriazi", "ekasoura", "prokopis"]:
                workspace.add_user(user)
            print("Users successfully linked to the workspace.")
    except Exception as e:
        print(f"User binding notice/error: {e}")

    print(f"\nΔιαβάζεται το Excel: {EXCEL_PATH}...")
    xls = pd.read_excel(EXCEL_PATH, sheet_name=list(SHEETS.keys()))

    print(f"Φόρτωση Hugging Face Dataset '{HF_REPO}' (για εικόνες)...")
    try:
        hf_dataset_pan = load_dataset(HF_REPO)
        hf_index_pan = {item["id"]: item for split in hf_dataset_pan.keys() for item in hf_dataset_pan[split]}
        print(f"Φορτώθηκαν {len(hf_index_pan)} εγγραφές από το Hugging Face.")
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση του HF dataset: {e}")
        hf_index_pan = {}

    for sheet_name, argilla_dataset_name in SHEETS.items():
        df = xls[sheet_name]
        print(f"\nΦύλλο '{sheet_name}': Βρέθηκαν {len(df)} εγγραφές για ανέβασμα.")
        if len(df) == 0: continue

        dataset = create_or_get_argilla_dataset(client, argilla_dataset_name, WORKSPACE_NAME)
        
        records = []
        for _, row in df.iterrows():
            q_id = clean_val(row.get("Question_ID"))
            model_name = clean_val(row.get("Model"))
            unique_record_id = f"{q_id}_{model_name}"

            clean_question = extract_clean_text(row.get("Input_Question"), "question")
            
            final_question_markdown = clean_question
            if argilla_dataset_name == "pass-or-fail-pan-eval":
                hf_item = hf_index_pan.get(q_id, {})
                images_rendered = images_to_markdown(hf_item.get("images", []))
                
                if images_rendered:
                    final_question_markdown = f"{images_rendered}\n\n**Ερώτηση:**\n{clean_question}"

            record = rg.Record(
                id=unique_record_id,
                fields={
                    "dataset": clean_val(row.get("Dataset")),
                    "subject": clean_val(row.get("Subject")),
                    "model_evaluated_field": model_name,
                    "question_id": q_id,
                    "input_question": final_question_markdown,
                    "reference_target": extract_clean_text(row.get("Reference_Target"), "reference"),
                    "model_answer": extract_clean_text(row.get("Model_Answer"), "answer"),
                    "human_grade": clean_val(row.get("Human_Grade")),
                    "human_explanation": clean_val(row.get("Human_Explanation")),
                    "gemma_score": clean_val(row.get("Gemma_Judge_Score")),
                    "gemma_explanation": clean_val(row.get("Gemma_Explanation")),
                    "mistral_score": clean_val(row.get("Mistral_Judge_Score")),
                    "mistral_explanation": clean_val(row.get("Mistral_Explanation")),
                    "bert_score": clean_val(row.get("BERTScore"))
                },
                metadata={
                    "model_evaluated_meta": model_name,
                    "subject_meta": clean_val(row.get("Subject"))
                }
            )
            records.append(record)
            
        print(f"Ανέβασμα εγγραφών στο '{argilla_dataset_name}'...")
        dataset.records.log(records)
        print("Ολοκληρώθηκε!")

if __name__ == "__main__":
    main()