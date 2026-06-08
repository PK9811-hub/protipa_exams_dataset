import datasets
import re
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

class GreekTokenizer:
    def tokenize(self, text):
        text = text.lower()
        return re.findall(r'[a-z0-9\u0370-\u03ff\u1f00-\u1fff]+', text)


ROUGE_SCORER = None


def process_results_gen(doc, results):
    completion = results[0]
    completion = re.sub(r'[*()\[\]"\']', '', completion).strip()
    
    gold_answer = doc.get("answer_text") or doc.get("answer") or ""
    gold_answer = str(gold_answer).strip()
    
    if "/" in gold_answer:
        true_refs = [a.strip() for a in gold_answer.split("/")]
    else:
        true_refs = [gold_answer]

    # BLEU (sacrebleu with international tokenization)
    bleu_scores = [bleu([[ref]], [completion]) for ref in true_refs]
    bleu_max = np.nanmax(bleu_scores)

    # ROUGE with custom Greek-safe tokenizer
    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeLsum"], 
            tokenizer=GreekTokenizer()
        )
    
    rouge_scores = []
    for ref in true_refs:
        scores = ROUGE_SCORER.score(ref, completion)
        rouge_scores.append({
            "rouge1": scores["rouge1"].fmeasure * 100.0,
            "rouge2": scores["rouge2"].fmeasure * 100.0,
            "rougeL": scores["rougeLsum"].fmeasure * 100.0,
        })

    rouge1_max = np.nanmax([s["rouge1"] for s in rouge_scores])
    rouge2_max = np.nanmax([s["rouge2"] for s in rouge_scores])
    rougeL_max = np.nanmax([s["rougeL"] for s in rouge_scores])

    # BERTScore using multilingual BERT to handle semantic similarity in Greek
    bertscore_f1_max = 0.0
    if BERTSCORE_AVAILABLE:
        # P, R, F1 are returned as tensors
        P, R, F1 = bert_score_fn(
            [completion]* len(true_refs),
            true_refs,
            lang="el",
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
        bertscore_f1_max = F1.max().item()

    return {
        "bleu_max": bleu_max,
        "rouge1_max": rouge1_max,
        "rouge2_max": rouge2_max,
        "rougeL_max": rougeL_max,
        "bertscore_f1_max": bertscore_f1_max,
    }


def bleu(refs, preds):
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def filter_by_mode_and_subject(dataset, mode='closed', subject=None):
    """
    Unified filter for evaluation mode and subject.
    - Closed Mode: MCQ, T/F, Matching, and Fill-in-the-gaps with choices.
    - Open Mode: Open-ended and Fill-in-the-gaps without choices.
    - Subject: Optional filtering by subject string.
    """
    def _filter_logic(x):
        # Optional subject filter
        if subject and x.get("subject") != subject:
            return False
            
        fmt = x.get("format")
        choices = x.get("choices")
        has_choices = isinstance(choices, list) and len(choices) > 0
        
        if mode == 'closed':
            return fmt in ["multiple_choice", "true_false", "matching"] or \
                   (fmt == "fill_in_the_gaps" and has_choices)
        elif mode == 'open':
            return fmt == "open_ended" or \
                   (fmt == "fill_in_the_gaps" and not has_choices)
        return True

    return dataset.filter(_filter_logic)

def doc_to_text_closed(doc):
    """
    MCQ prompt logic with specific Greek instructions for index-based answers.
    """
    prompt_parts = []
    
    # 1. Contextual Inputs
    if doc.get("input"): 
        prompt_parts.append(doc["input"])
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    # 2. Question
    prompt_parts.append(f"Ερώτηση: {doc['question']}")
    
    # 3. Choices (Index-based)
    if doc.get("choices"):
        prompt_parts.append("Επιλογές:")
        for i, choice in enumerate(doc["choices"]):
            prompt_parts.append(f"{i}. {choice}")
            
    # 4. Critical Instructions
    instruction = (
        "\n### ΟΔΗΓΙΑ ΜΟΡΦΟΠΟΙΗΣΗΣ (CRITICAL)\n"
        "Πρέπει να παρέχεις ΜΟΝΟ τον αριθμό του δείκτη (index) της σωστής επιλογής (π.χ. 0, 1, 2, 3...).\n"
        "ΠΡΟΣΟΧΗ: Ο αριθμός '2' που χρησιμοποιείται στα παραδείγματα παρακάτω είναι ΤΥΧΑΙΟΣ και αφορά μόνο τη ΜΟΡΦΗ της απάντησης εδώ.\n"
        "Η σωστή απάντηση εξαρτάται αποκλειστικά από την ερώτηση και μπορεί να είναι ΟΠΟΙΟΣΔΗΠΟΤΕ αριθμός.\n"
        "Μην επεξηγείς και μην γράφεις ολόκληρες προτάσεις.\n\n"
        "Παραδείγματα Μορφής:\n"
        "❌ ΛΑΘΟΣ: \"Η σωστή επιλογή είναι η 2.\"\n"
        "❌ ΛΑΘΟΣ: \"(2)\"\n"
        "✅ ΣΩΣΤΟ: 2 (ή 0 ή 1 ή 3... ανάλογα με τη σωστή επιλογή)\n\n"
        "Απάντηση:"
    )
    prompt_parts.append(instruction)
    return "\n".join(prompt_parts)

def doc_to_target(doc):
    """Extracts the integer index as the target string."""
    if doc.get("answer_index") is not None:
        return str(doc["answer_index"]).split(',')[0].strip()
    return ""

def doc_to_text_open(doc):
    """
    Prompt logic for open-ended and fill-in-the-gap questions.
    """
    prompt_parts = []
    format_type = doc.get("format")
    
    # 1. Instruction Engineering 
    if format_type == "open_ended":
        instruction = (
    "Απάντησε στην παρακάτω ερώτηση ανάπτυξης, δίνοντας μια ολοκληρωμένη και τεκμηριωμένη απάντηση.\n"
    "ΠΡΟΣΟΧΗ: Ξεκίνα την απάντησή σου απευθείας, χωρίς εισαγωγικές φράσεις, χωρίς να επαναλάβεις την ερώτηση και χωρίς χαιρετισμούς."
    )
        
    elif format_type == "fill_in_the_gaps":
        instruction = (
    "Γράψε ΜΟΝΟ τη σωστή λέξη ή τη σωστή φράση/τύπο που λείπει στην ερώτηση συμπλήρωσης κενών που σου δίνεται.\n"
    "ΚΡΙΣΙΜΗ ΟΔΗΓΙΑ: Μην δίνεις καμία απολύτως εξήγηση, μην γράφεις ολόκληρες προτάσεις, και μην χρησιμοποιείς εισαγωγικά.\n"
    "Η απάντησή σου πρέπει να περιέχει αποκλειστικά και ΜΟΝΟ τη λέξη ή φράση που συμπληρώνει το κενό."
    )
    
    prompt_parts.append(instruction)
    if doc.get("input"): 
        prompt_parts.append(f"Πλαίσιο/Κείμενο: {doc['input']}")
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    # 2. Question
    prompt_parts.append(f"Ερώτηση: {doc['question']}\n\nΑπάντηση:")
    
    return "\n\n".join(prompt_parts)

def doc_to_target_open(doc):
    """Extracts the expected text answer for open-ended evaluation."""
    ans = doc.get("answer_text") or doc.get("answer") or ""
    ans_str = str(ans).strip()

    if "/" in ans_str:
        return [a.strip() for a in ans_str.split("/")]
    
    return [ans_str]
# ------
def process_language_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='greek_language')
def process_maths_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='mathematics')
def process_religious_studies_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='religious studies')

def process_language_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='greek_language')
def process_maths_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='mathematics')
def process_physics_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='physics')
