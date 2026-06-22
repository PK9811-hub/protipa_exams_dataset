from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.solver import generate, system_message

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from evals.scorers import generic_judge_scorer

# Define strict, subject-specific grading rubrics for the judge
SUBJECT_RUBRICS = {
    "physics": (
        "You are a strict Greek school teacher grading a 12-year-old student's physics exam. "
        "Grade the student's submission based on the target solution (Criterion) using these strict rules:\n"
        "1. The answer must be written entirely in Greek. If the student uses any English terminology (e.g., 'Conduction', 'Convection', 'Radiation'), you must rate the answer 0.0.\n"
        "2. The student must use standard Greek school terminology. Specifically:\n"
        "   - The three heat transfer methods must be named: 'αγωγή', 'μεταφορά' or 'ρεύματα', and 'ακτινοβολία'.\n"
        "   - Using scientifically incorrect terminology or invented terms is strictly unacceptable and must result in a grade of 0.0.\n"
        "3. The answer must be concise and age-appropriate. If the answer is excessively verbose, contains unnecessary details, or is significantly longer than what is expected from a 12-year-old Greek student (comparing to the length of the Criterion), you must grade it 0.0.\n"
        "4. Rate 1.0 only if the physics explanation is correct, standard Greek terminology is used, AND the answer is concise/age-appropriate. Otherwise, rate 0.0."
    ),
    "mathematics": (
        "You are a strict Greek school teacher grading a 12-year-old student's mathematics exam. "
        "Verify that the student's final numerical answer matches the target solution (Criterion) exactly. "
        "If the final result is wrong or missing, rate 0.0. "
        "Rate 1.0 only if the derivation is mathematically sound and the final answer matches."
    ),
    "greek_language": (
        "You are a strict Greek school teacher grading a 12-year-old student's language exam. "
        "Grade the student's submission based on the target solution (Criterion). "
        "Ensure correct grammar, spelling, and vocabulary. "
        "Rate 1.0 if correct, 0.0 if incorrect."
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
    dataset_path: str = "ilsp/greek-protipa-exams",
    dataset_name: str | None = "default",
    split: str = "test",
    input_field: str = "question",
    context_field: str | None = "input",
    target_field: str = "answer_text",
    grading_instructions: str = "Compare the student's submission to the target solution. Rate 1.0 if correct, 0.0 if incorrect.",
    filter_field: str | None = None,
    filter_value: str | None = None,
    grader_model: str | None = None
):
    """
    A generic Inspect AI task that loads any Hugging Face dataset,
    filters it, and applies dynamic subject-specific grading rubrics and system prompts.
    """
    import json
    
    def record_to_sample(x):
        subject = x.get("subject")
        system_instruction = SUBJECT_SYSTEM_INSTRUCTIONS.get(subject, "Είσαι ένας Έλληνας μαθητής που απαντά σε διαγώνισμα.")
        
        user_prompt_parts = []
        if context_field and x.get(context_field):
            user_prompt_parts.append(f"Context: {x.get(context_field)}")
            
        if x.get("image_description"):
            user_prompt_parts.append(f"[Image Description: {x.get('image_description')}]")
            
        if x.get("image_transcription"):
            user_prompt_parts.append(f"[Image Transcription: {x.get('image_transcription')}]")
            
        user_prompt_parts.append(f"Question: {x.get(input_field)}")
        user_input = "\n\n".join(user_prompt_parts)
        
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
            input=user_input,
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
