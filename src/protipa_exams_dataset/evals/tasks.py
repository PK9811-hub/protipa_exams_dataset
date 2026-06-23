from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.solver import generate, system_message

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from evals.scorers import generic_judge_scorer

# Define strict, subject-specific grading rubrics for the judge
#SUBJECT_RUBRICS = {
    #"physics": (
        #"You are a strict Greek school teacher grading a 12-year-old student's physics exam. "
        #"Grade the student's submission based on the target solution (Criterion) using these strict rules:\n"
        #"1. The answer must be written entirely in Greek. If the student uses any English terminology (e.g., 'Conduction', 'Convection', 'Radiation'), you must rate the answer 0.0.\n"
        #"2. The student must use standard Greek school terminology. Specifically:\n"
        #"   - The three heat transfer methods must be named: 'αγωγή', 'μεταφορά' or 'ρεύματα', and 'ακτινοβολία'.\n"
        #"   - Using scientifically incorrect terminology or invented terms is strictly unacceptable and must result in a grade of 0.0.\n"
        #"3. The answer must be concise and age-appropriate. If the answer is excessively verbose, contains unnecessary details, or is significantly longer than what is expected from a 12-year-old Greek student (comparing to the length of the Criterion), you must grade it 0.0.\n"
        #"4. Rate 1.0 only if the physics explanation is correct, standard Greek terminology is used, AND the answer is concise/age-appropriate. Otherwise, rate 0.0."
    #),
    #"mathematics": (
        #"You are a strict Greek school teacher grading a 12-year-old student's mathematics exam. "
        #"Verify that the student's final numerical answer matches the target solution (Criterion) exactly. "
        #"If the final result is wrong or missing, rate 0.0. "
        #"Rate 1.0 only if the derivation is mathematically sound and the final answer matches."
    #),
    #"greek_language": (
        #"You are a strict Greek school teacher grading a 12-year-old student's language exam. "
        #"Grade the student's submission based on the target solution (Criterion). "
        #"Ensure correct grammar, spelling, and vocabulary. "
        #"Rate 1.0 if correct, 0.0 if incorrect."
    #)
#}
# Define strict, subject-specific grading rubrics for the judge (Translated & Granular)
#SUBJECT_RUBRICS = {
    #"physics": (
        #"Είσαι ένας αυστηρός Έλληνας εκπαιδευτικός που βαθμολογεί το γραπτό Φυσικής ενός 12χρονου μαθητή. "
        #"Αξιολόγησε την απάντηση του μαθητή (Submission) συγκρίνοντάς τη με την πρότυπη λύση (Criterion).\n"
        #"Χρησιμοποίησε κλίμακα βαθμολόγησης: 0.0, 0.25, 0.5, 0.75, ή 1.0.\n"
        #"Κανόνες:\n"
        #"1. Η απάντηση πρέπει να είναι εξ ολοκλήρου στα Ελληνικά. Αν χρησιμοποιηθεί αγγλική ορολογία (π.χ. 'Conduction'), ο βαθμός είναι 0.0.\n"
        #"2. Πρέπει να χρησιμοποιείται η σωστή σχολική ορολογία (π.χ. 'αγωγή', 'μεταφορά' ή 'ρεύματα', 'ακτινοβολία').\n"
        #"3. Δώσε μερική βαθμολογία (0.25 - 0.75) αν ο μαθητής έχει κατανοήσει το φυσικό φαινόμενο, αλλά κάνει μικρά λάθη στην εξήγηση ή παραλείπει μια λεπτομέρεια.\n"
        #"4. Βάλε 1.0 μόνο αν η εξήγηση είναι απόλυτα σωστή, περιεκτική και χρησιμοποιεί τη σωστή ορολογία."
    #),
    #"mathematics": (
        #"Είσαι ένας αυστηρός Έλληνας εκπαιδευτικός που βαθμολογεί το γραπτό Μαθηματικών ενός 12χρονου μαθητή. "
        #"Αξιολόγησε την απάντηση του μαθητή (Submission) συγκρίνοντάς τη με την πρότυπη λύση (Criterion).\n"
        #"Χρησιμοποίησε κλίμακα βαθμολόγησης: 0.0, 0.25, 0.5, 0.75, ή 1.0.\n"
        #"Κανόνες:\n"
        #"1. Δώσε 1.0 αν η μεθοδολογία είναι σωστή και το τελικό αποτέλεσμα ταυτίζεται απόλυτα με το Criterion.\n"
        #"2. Δώσε μερική βαθμολογία (π.χ. 0.5 ή 0.75) αν ο μαθητής ακολούθησε τα σωστά βήματα ή βρήκε μέρος της λύσης (π.χ. βρήκε τη μία από τις δύο λύσεις μιας εξίσωσης), αλλά έκανε λάθος στις τελικές πράξεις.\n"
        #"3. Δώσε 0.0 αν η λογική είναι εντελώς λανθασμένη ή δεν υπάρχει τελικό αποτέλεσμα."
    #),
    #"greek_language": (
        #"Είσαι ένας αυστηρός Έλληνας εκπαιδευτικός που βαθμολογεί το γραπτό Νεοελληνικής Γλώσσας ενός 12χρονου μαθητή. "
        #"Αξιολόγησε την απάντηση του μαθητή (Submission) συγκρίνοντάς τη με την πρότυπη λύση (Criterion).\n"
        #"Χρησιμοποίησε κλίμακα βαθμολόγησης: 0.0, 0.25, 0.5, 0.75, ή 1.0.\n"
        #"Κανόνες:\n"
        #"1. Εστίασε στην ορθογραφία, τη γραμματική, το συντακτικό και την απόδοση του σωστού νοήματος.\n"
        #"2. Δώσε 1.0 αν η απάντηση είναι άψογη νοηματικά και συντακτικά.\n"
        #"3. Δώσε μερική βαθμολογία (0.25 - 0.75) αν ο μαθητής βρήκε το σωστό νόημα ή τη σωστή λέξη (π.χ. ένα συνώνυμο), αλλά έκανε συντακτικά λάθη, δημιούργησε πλεονασμούς στην πρόταση, ή δεν ολοκλήρωσε σωστά την άσκηση.\n"
        #"4. Δώσε 0.0 αν η απάντηση είναι εντελώς εκτός θέματος."
    #)
#}
# Define strict, subject-specific grading rubrics for the judge (Translated & Granular)
SUBJECT_RUBRICS = {
    "physics": (
        "Είσαι ένας αυστηρός Έλληνας εκπαιδευτικός που βαθμολογεί το γραπτό Φυσικής ενός 12χρονου μαθητή. "
        "Αξιολόγησε την απάντηση του μαθητή (Submission) συγκρίνοντάς τη με την πρότυπη λύση (Criterion).\n"
        "Χρησιμοποίησε κλίμακα βαθμολόγησης: 0.0, 0.25, 0.5, 0.75, ή 1.0.\n"
        "Κανόνες:\n"
        "1. Η απάντηση πρέπει να είναι εξ ολοκλήρου στα Ελληνικά. Αν χρησιμοποιηθεί αγγλική ορολογία (π.χ. 'Conduction'), ο βαθμός είναι 0.0.\n"
        "2. Δώσε 1.0 μόνο αν η εξήγηση είναι απόλυτα σωστή, περιεκτική και χρησιμοποιεί τη σωστή σχολική ορολογία.\n"
        "3. Δώσε 0.75 αν η εξήγηση είναι σωστή αλλά λείπει μια μικρή λεπτομέρεια.\n"
        "4. Δώσε 0.50 αν ο μαθητής έχει κατανοήσει το φαινόμενο, αλλά κάνει κάποιο σημαντικό λάθος στην περιγραφή ή παραλείπει κρίσιμα δεδομένα.\n"
        "5. Δώσε 0.25 αν η απάντηση είναι κυρίως λάθος, αλλά δείχνει μια πολύ βασική, μερική κατανόηση του φαινομένου ή χρησιμοποιεί έστω έναν σωστό όρο.\n"
    ),
    "mathematics": (
        "Είσαι ένας αυστηρός Έλληνας εκπαιδευτικός που βαθμολογεί το γραπτό Μαθηματικών ενός 12χρονου μαθητή. "
        "Αξιολόγησε την απάντηση του μαθητή (Submission) συγκρίνοντάς τη με την πρότυπη λύση (Criterion).\n"
        "Χρησιμοποίησε κλίμακα βαθμολόγησης: 0.0, 0.25, 0.5, 0.75, ή 1.0.\n"
        "Κανόνες:\n"
        "1. Δώσε 1.0 αν η μεθοδολογία είναι σωστή και το τελικό αποτέλεσμα ταυτίζεται απόλυτα με το Criterion.\n"
        "2. Δώσε 0.75 αν η μεθοδολογία είναι ολόσωστη αλλά υπάρχει ένα μικρό αριθμητικό λάθος στο τελικό αποτέλεσμα.\n"
        "3. Δώσε 0.50 αν ο μαθητής ακολούθησε τα σωστά βήματα μέχρι τη μέση ή βρήκε μόνο μέρος της λύσης (π.χ. τη μία από τις δύο λύσεις μιας εξίσωσης).\n"
        "4. Δώσε 0.25 αν η μεθοδολογία είναι λανθασμένη ή ατελής, αλλά εφάρμοσε σωστά κάποιον βασικό τύπο ή έκανε μια σωστή αρχική σκέψη.\n"
        "5. Δώσε 0.0 αν και η λογική και το αποτέλεσμα είναι εντελώς λανθασμένα ή δεν υπάρχει καμία προσπάθεια λύσης."
    ),
    "greek_language": (
        "Είσαι ένας αυστηρός Έλληνας εκπαιδευτικός που βαθμολογεί το γραπτό Νεοελληνικής Γλώσσας ενός 12χρονου μαθητή. "
        "Αξιολόγησε την απάντηση του μαθητή (Submission) συγκρίνοντάς τη με την πρότυπη λύση (Criterion).\n"
        "Χρησιμοποίησε κλίμακα βαθμολόγησης: 0.0, 0.25, 0.5, 0.75, ή 1.0.\n"
        "Κανόνες:\n"
        "1. Εστίασε στην ορθογραφία, τη γραμματική, το συντακτικό και την απόδοση του σωστού νοήματος.\n"
        "2. Δώσε 1.0 αν η απάντηση είναι άψογη νοηματικά και συντακτικά.\n"
        "3. Δώσε 0.75 αν βρήκε το σωστό νόημα ή συνώνυμο, αλλά έκανε ένα ελαφρύ συντακτικό ή ορθογραφικό λάθος.\n"
        "4. Δώσε 0.50 αν βρήκε μέρος της απάντησης ή αν η επιλογή του δημιούργησε πλεονασμούς και άκομψες προτάσεις στο κείμενο.\n"
        "5. Δώσε 0.25 αν η απάντηση είναι ελλιπής ή μερικώς εκτός θέματος, αλλά περιέχει τουλάχιστον ένα σωστό στοιχείο ή λέξη.\n"
        "6. Δώσε 0.0 αν η απάντηση είναι εντελώς εκτός θέματος ή λανθασμένη."
    )
}

# Define strict, subject-specific constraints for the student model (under evaluation)
SUBJECT_SYSTEM_INSTRUCTIONS = {
    "physics": (
        "Είσαι ένας 12χρονος Έλληνας μαθητής που απαντά σε διαγώνισμα Φυσικής.\n"
        "Απάντησε στην ερώτηση σύντομα, απλά και με σαφήνεια (1-3 προτάσεις το πολύ).\n"
        "Χρησιμοποίησε αποκλειστικά την επίσημη ελληνική σχολική ορολογία της Φυσικής (π.χ. 'αγωγή', 'μεταφορά' ή 'ρεύματα', 'ακτινοβολία').\n"
        "Απαγορεύεται αυστηρά η χρήση αγγλικών όρων (π.χ. Conduction, Convection, Radiation) ή μη-δόκιμων μεταφράσεων."
    ),
    "mathematics": (
        "Είσαι ένας 12χρονος Έλληνας μαθητής που απαντά σε διαγώνισμα Μαθηματικών.\n"
        "Λύσε το πρόβλημα βήμα-βήμα, δείχνοντας τις πράξεις σου απλά, και γράψε το τελικό αριθμητικό αποτέλεσμα καθαρά στο τέλος."
    ),
    "greek_language": (
        "Είσαι ένας 12χρονος Έλληνας μαθητής που απαντά σε διαγώνισμα Νεοελληνικής Γλώσσας.\n"
        "Γράψε την απάντησή σου σύντομα, με σωστή γραμματική, ορθογραφία και σύνταξη στα ελληνικά."
    )
}

@task
def generic_evaluation(
    dataset_path: str = "ilsp/greek-protipa-exams-private",
    dataset_name: str | None = "default",
    split: str = "test",
    fewshot_split: str = "dev",  
    num_fewshot: int = 5,        
    input_field: str = "question",
    context_field: str | None = "input",
    target_field: str = "answer_text",
    grading_instructions: str = "Σύγκρινε την απάντηση του μαθητή με την πρότυπη λύση. Βαθμολόγησε αυστηρά στα Ελληνικά χρησιμοποιώντας κλίμακα 0.0, 0.25, 0.5, 0.75, ή 1.0 ανάλογα με την ορθότητα.",
    filter_field: str | None = None,
    filter_value: str | None = None,
    grader_model: str | None = None
):
    """
    A generic Inspect AI task that loads any Hugging Face dataset,
    filters it, and applies dynamic subject-specific grading rubrics, system prompts, and few-shot examples.
    """
    import json
    from datasets import load_dataset  

    # 1. Φορτώνουμε το dev split ΜΙΑ φορά στην αρχή του task για ταχύτητα
    try:
        dev_data = load_dataset(dataset_path, dataset_name, split=fewshot_split)
        dev_records = list(dev_data)
    except Exception as e:
        print(f"Warning: Could not load few-shot split '{fewshot_split}'. Proceeding with 0 shots.")
        dev_records = []

    def record_to_sample(x):
        subject = x.get("subject")
        format_type = x.get("format")
        system_instruction = SUBJECT_SYSTEM_INSTRUCTIONS.get(subject, "Είσαι ένας Έλληνας μαθητής που απαντά σε διαγώνισμα.")
        
        user_prompt_parts = []
        if context_field and x.get(context_field):
            user_prompt_parts.append(f"Context: {x.get(context_field)}")
            
        if x.get("image_description"):
            user_prompt_parts.append(f"[Image Description: {x.get('image_description')}]")
            
        if x.get("image_transcription"):
            user_prompt_parts.append(f"[Image Transcription: {x.get('image_transcription')}]")
            
        user_prompt_parts.append(f"Question: {x.get(input_field)}")
        core_question = "\n\n".join(user_prompt_parts)
        
        # --- 2. FEW SHOT LOGIC ---
        final_user_input = core_question
        if num_fewshot > 0 and dev_records:
            # Βρίσκουμε παραδείγματα που ταιριάζουν στο ίδιο μάθημα (subject) και τύπο (format)
            matching_shots = [r for r in dev_records if r.get("subject") == subject and r.get("format") == format_type]
            
            if matching_shots:
                few_shot_text = "Ακολουθούν μερικά παραδείγματα προς διευκόλυνσή σου:\n\n"
                # Παίρνουμε τα πρώτα 'num_fewshot' παραδείγματα
                for i, shot in enumerate(matching_shots[:num_fewshot]):
                    q = shot.get(input_field, "")
                    a = shot.get(target_field, "")
                    
                    # Αν το παράδειγμα έχει context, το προσθέτουμε
                    shot_context = f"Context: {shot.get(context_field)}\n" if context_field and shot.get(context_field) else ""
                    
                    few_shot_text += f"--- Παράδειγμα {i+1} ---\n{shot_context}Question: {q}\nΑπάντηση: {a}\n\n"
                
                few_shot_text += "--- Τέλος Παραδειγμάτων ---\n\nΤώρα απάντησε στην παρακάτω ερώτηση:\n"
                
                # Ενώνουμε τα παραδείγματα με την τωρινή ερώτηση
                final_user_input = few_shot_text + core_question
        
        target = x.get(target_field) or ""
        sample_rubric = SUBJECT_RUBRICS.get(subject, grading_instructions)
        
        metadata = {}
        for k, v in x.items():
            if k not in [input_field, context_field, target_field, "image"]:
                try:
                    json.dumps(v)
                    metadata[k] = v
                except (TypeError, OverflowError):
                    pass
        metadata["grading_instructions"] = sample_rubric
        metadata["system_instruction"] = system_instruction
        
        return Sample(
            input=final_user_input,
            target=str(target),
            id=str(x.get("id", "")),
            metadata=metadata
        )

    raw_dataset = hf_dataset(
        path=dataset_path,
        name=dataset_name,
        split=split,
        sample_fields=record_to_sample
    )
    
    # Materialize samples and purge non-JSON-serializable fields (like PIL Images) from metadata
    samples = []
    for sample in raw_dataset:
        if sample.metadata:
            cleaned_metadata = {}
            for k, v in sample.metadata.items():
                try:
                    json.dumps(v)
                    cleaned_metadata[k] = v
                except (TypeError, OverflowError):
                    pass
            sample.metadata = cleaned_metadata
        samples.append(sample)
    dataset = MemoryDataset(samples)
    
    if filter_field and filter_value:
        fields = [f.strip() for f in filter_field.split(";")]
        values = [v.strip() for v in filter_value.split(";")]
        
        for f_name, f_val in zip(fields, values):
            if f_name == "has_image_description":
                dataset = dataset.filter(
                    lambda sample, val=f_val: bool(sample.metadata.get("image_description")) == (val.lower() == "true")
                )
            else:
                val_str = str(f_val).strip("[]()'\"")
                allowed_values = [v.strip().strip("'\"") for v in val_str.split(",")]
                dataset = dataset.filter(
                    lambda sample, name=f_name, vals=allowed_values: str(sample.metadata.get(name)).strip() in vals
                )
        
    return Task(
        dataset=dataset,
        plan=[
            system_message("{system_instruction}"),
            generate()
        ],
        scorer=generic_judge_scorer(instructions=grading_instructions, model=grader_model),
        model_roles={"grader": grader_model} if grader_model else None
    )