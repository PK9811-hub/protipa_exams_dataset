import datasets

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
    
    # 1. Instruction
    instruction = (
        "Δίνεται η παρακάτω ερώτηση από σχολικές εξετάσεις.\n"
        "Αν είναι ερώτηση ανάπτυξης, δώσε μια ολοκληρωμένη και τεκμηριωμένη απάντηση.\n"
        "Αν είναι ερώτηση συμπλήρωσης κενών, γράψε τη σωστή λέξη ή τη σωστή φράση που λείπει."
    )
    prompt_parts.append(instruction)
    
    # 2. Contextual Inputs 
    if doc.get("input"): 
        prompt_parts.append(f"Πλαίσιο/Κείμενο: {doc['input']}")
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    # 3. Question
    prompt_parts.append(f"Ερώτηση: {doc['question']}\n\nΑπάντηση:")
    
    return "\n\n".join(prompt_parts)

def doc_to_target_open(doc):
    """Extracts the expected text answer for open-ended evaluation."""
    ans = doc.get("answer_text") or doc.get("answer") or ""
    return [str(ans).strip()]

# ------
def process_language_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='greek_language')
def process_maths_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='mathematics')
def process_religious_studies_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='religious studies')

def process_language_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='greek_language')
def process_maths_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='mathematics')
def process_physics_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='physics')