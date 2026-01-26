import random
import json
import ast
import logging
import os
import pandas as pd
import ast
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

def filter_dataset(dataset, mode='closed'):
    """
    Final cleaned version of filter_dataset.
    Filters based on Subject and Format logic (Open vs Closed).
    """
    filtered = []
    
    valid_subjects = [
        'greek_language', 'mathematics', 'physics', 'religious studies',
        'ΓΛΩΣΣΑ', 'ΜΑΘΗΜΑΤΙΚΑ', 'ΦΥΣΙΚΗ', 'ΘΡΗΣΚΕΥΤΙΚΑ'
    ]
    
    print(f"🔍 Filtering for mode: {mode.upper()}...")

    for item in dataset:
        # 1. Ανάκτηση δεδομένων
        subj = str(item.get('subject', '')).strip()
        fmt = str(item.get('format', '')).strip()
        
        # 2. Ασφαλής ανάκτηση choices
        raw_choices = item.get('choices', [])
        
        if isinstance(raw_choices, list):
            choices = raw_choices
        elif isinstance(raw_choices, str):
            try:
                choices = ast.literal_eval(raw_choices)
            except:
                choices = []
        else:
            choices = []
            
        if choices is None: choices = []
        
        # 3. Έλεγχος Μαθήματος
        is_valid_subj = any(s.lower() == subj.lower() for s in valid_subjects)
        if not is_valid_subj:
            continue

        # 4. ΛΟΓΙΚΗ ΔΙΑΧΩΡΙΣΜΟΥ
        has_choices = (len(choices) > 0)
        should_keep = False

        if mode == 'closed':
            # Κλειστά formats + Fill-in με επιλογές
            if fmt in ['multiple_choice', 'true_false', 'matching']:
                should_keep = True
            elif fmt == 'fill_in_the_gaps' and has_choices:
                should_keep = True
                
        elif mode == 'open':
            # Ανοιχτά formats + Fill-in χωρίς επιλογές
            if fmt == 'open_ended':
                should_keep = True
            elif fmt == 'fill_in_the_gaps' and not has_choices:
                should_keep = True

        if should_keep:
            filtered.append(item)

    print(f"✅ Found {len(filtered)} items for mode '{mode}'.")
    return filtered
    

def load_protipa_dataset(repo_id=None, split=None):
    """
    Loads and concatenates train and test splits if split is not specified.
    """
    if repo_id is None:
        repo_id = os.getenv("HF_REPO_ID", "PK9811-hub/protipa_exams_dataset")
    
    logger.info(f"Loading dataset from Hugging Face: {repo_id}")
    dataset_dict = load_dataset(repo_id)
    
    if split:
        return dataset_dict[split]
    
    # default to concatenating train and test
    return concatenate_datasets([dataset_dict['train'], dataset_dict['test']])

def process_matching_row(row):
    """
    Processes a single matching exercise row to create a list of answer options
    (shuffled versions of the matching) and identifying the correct index.
    """
    answer_raw = row.get('answer')
    if not answer_raw:
        return None
    
    try:
        # 1. Parse answer if it's a string
        if isinstance(answer_raw, str):
            # Try to handle common formats like '["A-1", "B-2"]' or "['A-1', 'B-2']"
            try:
                correct_list = json.loads(answer_raw.replace("'", '"'))
            except:
                try:
                    correct_list = ast.literal_eval(answer_raw)
                except:
                    # Fallback if it's just a raw CSV string
                    if '-' in answer_raw or ':' in answer_raw:
                        correct_list = [s.strip() for s in answer_raw.split(',')]
                    else:
                        return None
        else:
            correct_list = answer_raw
            
        if not isinstance(correct_list, list) or len(correct_list) == 0:
            return None
            
        # 2. Extract pairs. Assume format "SideA-SideB" or similar separator
        # We'll try to find the separator (usually '-' or ':')
        first_item = str(correct_list[0])
        sep = '-' if '-' in first_item else ':' if ':' in first_item else None
        
        if not sep:
            return None
            
        left_side = []
        right_side = []
        for pair in correct_list:
            parts = str(pair).split(sep, 1)
            if len(parts) == 2:
                left_side.append(parts[0])
                right_side.append(parts[1])
            else:
                # If one item doesn't have the sep, we can't reliably shuffle
                return None
        
        # 3. Generate 3 distractors
        distractors = []
        max_attempts = 100
        while len(distractors) < 3 and max_attempts > 0:
            shuffled_right = random.sample(right_side, len(right_side))
            candidate = [f"{l}{sep}{r}" for l, r in zip(left_side, shuffled_right)]
            
            # Check if it's actually different and not already in distractors
            if candidate != correct_list and candidate not in distractors:
                distractors.append(candidate)
            max_attempts -= 1
            
        # 4. Create the final list of options (a list of lists)
        options = [correct_list] + distractors
        
        # 5. Shuffle the list of options
        random.shuffle(options)
        
        # 6. Find the new index of the correct answer
        correct_index = options.index(correct_list)
        
        return pd.Series({
            'processed_choices': options,
            'new_answer_index': correct_index
        })
        
    except Exception as e:
        logger.error(f"Error processing matching row: {e}")
        return None

def apply_matching_processing(df, target_ids=None):
    """
    Applies matching processing to a whole dataframe using .loc to avoid warnings.
    If target_ids is provided, it selectively processes only those unique_ids.
    Returns the full input dataframe with processed matches updated.
    """
    if df.empty:
        return df.copy()

    # Create a copy to work on
    df = df.copy()
    
    # Initialize the new columns using .loc if they don't exist
    for col in ['processed_choices', 'new_answer_index']:
        if col not in df.columns:
            df.loc[:, col] = None
    
    # Determine which rows to process
    rows_to_process = df
    if target_ids is not None and 'unique_id' in df.columns:
        rows_to_process = df[df['unique_id'].isin(target_ids)]
    
    if rows_to_process.empty:
        return df

    # Compute updates
    try:
        updates = rows_to_process.apply(process_matching_row, axis=1, result_type='expand')
        
        if isinstance(updates, pd.DataFrame) and 'processed_choices' in updates.columns:
            valid_mask = updates['processed_choices'].notnull()
            
            if valid_mask.any():
                # The index of updates matches the index of df
                valid_indices = updates.index[valid_mask]
                
                df.loc[valid_indices, ['processed_choices', 'new_answer_index']] = updates.loc[valid_indices]
                df.loc[valid_indices, 'answer'] = updates.loc[valid_indices, 'processed_choices']
                df.loc[valid_indices, 'answer_index'] = updates.loc[valid_indices, 'new_answer_index']
    except Exception as e:
        logger.error(f"Error in apply_matching_processing: {e}")
    
    return df

def clean_dataset_paths(df):
    """
    Cleans path-related columns (like 'image' and 'source_file') to keep only the basename.
    """
    df = df.copy()
    
    # helper to handle list of paths or single path
    def get_basename(path_val):
        if isinstance(path_val, list):
            return [os.path.basename(p.replace('\\', '/')) for p in path_val]
        if isinstance(path_val, str):
            return os.path.basename(path_val.replace('\\', '/'))
        return path_val

    if 'image' in df.columns:
        df.loc[:, 'image'] = df['image'].apply(get_basename)
    
    if 'source_file' in df.columns:
        df.loc[:, 'source_file'] = df['source_file'].apply(get_basename)
        
    return df

def process_results_open(doc, results):
    """
    Προετοιμάζει τα δεδομένα για τις μετρικές BLEU/ChrF στα Open-Ended tasks.
    Τοποθετεί το Ground Truth μέσα σε λίστα [] (list of lists) γιατί έτσι 
    απαιτούν οι βιβλιοθήκες 'sacrebleu'/'evaluate' για να μην βγάλουν 0.0.
    """
    # Η απάντηση που έδωσε το μοντέλο (String)
    # Το lm-eval επιστρέφει λίστα, παίρνουμε το πρώτο στοιχείο
    completion = results[0]
    
    # Η σωστή απάντηση (String) από το dataset
    target = doc["answer"]
    
    # Επιστρέφουμε ένα λεξικό που αντιστοιχεί κάθε μετρική 
    # στη μορφή: (prediction, reference)
    return {
        "bleu": (completion, [target]),  
        "chrf": (completion, [target])
    }