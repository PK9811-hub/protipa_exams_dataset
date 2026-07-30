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
        
        subj = str(item.get('subject', '')).strip()
        fmt = str(item.get('format', '')).strip()
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
        
        is_valid_subj = any(s.lower() == subj.lower() for s in valid_subjects)
        if not is_valid_subj:
            continue

        has_choices = (len(choices) > 0)
        should_keep = False

        if mode == 'closed':
            if fmt in ['multiple_choice', 'true_false', 'matching']:
                should_keep = True
            elif fmt == 'fill_in_the_gaps' and has_choices:
                should_keep = True
                
        elif mode == 'open':
            if fmt == 'open_ended':
                should_keep = True
            elif fmt == 'fill_in_the_gaps' and not has_choices:
                should_keep = True

        if should_keep:
            if fmt == 'true_false':
                ans_idx = item.get('answer_index')
                if ans_idx is None or str(ans_idx).lower() == 'nan':
            
                    ans_text = str(item.get('answer_text', '') or item.get('answer', '')).lower()
                    
                    if 'σωστό' in ans_text or 'true' in ans_text:
                        item['answer_index'] = 0
                    elif 'λάθος' in ans_text or 'false' in ans_text:
                        item['answer_index'] = 1
            
            filtered.append(item)

    print(f"✅ Found {len(filtered)} items for mode '{mode}'.")
    return filtered
    

def load_protipa_dataset(repo_id=None, split=None):
    """
    Loads and concatenates train and test splits if split is not specified.
    """
    if repo_id is None:
        repo_id = os.getenv("HF_REPO_ID")
    
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
    answer_raw = row.get('answer_text')
    if not answer_raw:
        answer_raw = row.get('answer')
        if not answer_raw:
            return None
    
    try:
        # 1. Parse answer if it's a string
        if isinstance(answer_raw, str):
            if '-' in answer_raw or ':' in answer_raw:
                correct_list = [s.strip() for s in answer_raw.split(',')]
            # Try to handle common formats like '["A-1", "B-2"]' or "['A-1', 'B-2']"
            else:
                try:
                    correct_list = json.loads(answer_raw.replace("'", '"'))
                except:
                    try:
                        correct_list = ast.literal_eval(answer_raw)
                    except:
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
        raw_options = [correct_list] + distractors
        
        # 5. Shuffle the list of options
        random.shuffle(raw_options)
        
        options = [", ".join(opt) for opt in raw_options]
        
        # 6. Find the new index of the correct answer
        correct_string = ", ".join(correct_list)
        correct_index = options.index(correct_string)
        
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
    mask = df['format'].astype(str).str.contains('matching', case=False, na=False)
    
    if target_ids is not None and 'unique_id' in df.columns:
        mask = mask & df[df['unique_id'].isin(target_ids)]
    
    rows_to_process = df[mask]
    
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
                
                #df.loc[valid_indices, ['processed_choices', 'new_answer_index']] = updates.loc[valid_indices]
                df.loc[valid_indices, 'choices'] = updates.loc[valid_indices, 'processed_choices']
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
    Prepares the data for BLEU/ChrF metrics in Open-Ended tasks. Places the Ground Truth inside a list [] (list of lists) because that's how the 'sacrebleu'/'evaluate' libraries require it to avoid returning 0.0.
    """
    completion = results[0]
    
    target = doc["answer"]
    
    return {
        "bleu": (completion, [target]),  
        "chrf": (completion, [target])
    }

def process_results_bypass(doc, results):
    """
    Simple function that returns data for exact_match.
    Purpose: To avoid BLEU crash in lm-eval.
    """
    completion = results[0]
    target = doc["answer_text"]
    
    return {
        "exact_match": 0.0 
    }