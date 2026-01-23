import os
import json
import re
import pandas as pd
import argparse
from pathlib import Path
import ast
import random
from datasets import Dataset, Image, Features, Sequence
from huggingface_hub import create_repo, repo_exists, HfApi
from dotenv import load_dotenv, find_dotenv

# Initialize dotenv
load_dotenv(find_dotenv())

# --- HELPER FUNCTIONS (Derived from Dataset Creation Utilities) ---

def parse_filename_metadata(filename):
    name_no_ext = os.path.splitext(filename)[0]
    pattern_full = r"ΘΕΜΑΤΑ_([^_]+)_([^_]+)_(\d{4})_(\d+)"
    match = re.match(pattern_full, name_no_ext)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4)
    pattern_short = r"ΘΕΜΑΤΑ_([^_]+)_([^_]+)_(\d{4})"
    match_short = re.match(pattern_short, name_no_ext)
    if match_short:
        return match_short.group(1), match_short.group(2), match_short.group(3), "1"
    return None, None, None, None

def parse_markdown_answers(md_file):
    answers_dict = {}
    if not os.path.exists(md_file):
        return answers_dict
    with open(md_file, "r", encoding="utf-8-sig") as f:
        content = f.read()
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith("---"): continue
            match = re.match(r"^(\S+)\s+(.*)", line)
            if match:
                q_id = match.group(1).strip() # Fixed: removed replace(".", "")
                ans_body = match.group(2).strip()
                if len(ans_body) >= 2 and ans_body.startswith('"') and ans_body.endswith('"'):
                    ans_body = ans_body[1:-1]
                if q_id not in answers_dict:
                    answers_dict[q_id] = []
                answers_dict[q_id].append(ans_body)
    return answers_dict

def detect_exercise_type(q_text, q_choices):
    text = q_text.lower() if q_text else ""
    choices_list = q_choices if q_choices else []
    choices_text_raw = " ".join([str(c) for c in choices_list])
    
    is_matching_keyword = "αντιστοιχ" in text or "στήλη" in text
    has_upper = any(re.match(r'^[Α-ΩA-Z]\.', str(c).strip()) for c in choices_list)
    has_numbers = any(re.match(r'^\d+\.', str(c).strip()) for c in choices_list)
    has_lower = any(re.match(r'^([α-ωa-z]|στ)\.', str(c).strip()) for c in choices_list)
    is_matching_structure = (has_upper and has_numbers) or (has_lower and has_numbers) or (has_upper and has_lower)

    if is_matching_keyword or is_matching_structure: return "matching"

    gap_keywords = ["κενό", "κενά", "____", "__________"]
    is_gap = any(k in text for k in gap_keywords)
    if is_gap and not (has_upper or has_numbers or has_lower): return "fill-in-the-gaps"

    if not choices_list: return "open"
    
    tf_keywords = ["Σωστό", "Λάθος", "σωστό", "λάθος", "σωστή", "λανθασμένη", "Σ/Λ"]
    if len(choices_list) <= 2 and any(kw in choices_text_raw for kw in tf_keywords):
        return "true/false"
    
    if len(choices_list) >= 2: return "multiple choice"
    return "open"

def unify_label(text):
    if not text: return ""
    clean = str(text).replace(".", "").replace(")", "").replace("(", "").strip().upper()
    mapping = {'Α': 'A', 'Β': 'B', 'Γ': 'C', 'Δ': 'D', 'Ε': 'E', 'Ζ': 'F', 'Η': 'G', 'Θ': 'H'}
    return mapping.get(clean, clean)

def find_answer_index(choices, answer_raw):
    if not choices or not answer_raw: return None
    ans_str = str(answer_raw).strip()
    ans_unified = unify_label(ans_str)
    for idx, choice in enumerate(choices):
        choice_str = str(choice).strip()
        if choice_str == ans_str: return idx
        match = re.match(r"^.*?([\(\d\w]{1,3})[\.\)\s]\s*(.*)", choice_str)
        if match and unify_label(match.group(1)) == ans_unified: return idx
    return None

def parse_image_txt(txt_path):
    if not os.path.exists(txt_path): return None, None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "Transcription" in content:
            parts = content.split("Transcription")
            description = parts[0].replace("Description", "").strip()
            transcription = parts[1].strip()
        else:
            description = content.replace("Description", "").strip()
            transcription = None
        return description, transcription
    except: return None, None

def extract_points(text):
    """Extracts numeric value from strings like 'Μονάδες 2.5' or 'Μονάδες 10'."""
    if not text: return None
    # Regular expression to find integers or floats
    match = re.search(r"(\d+(?:\.\d+)?)", str(text))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def transform_matching_to_mc(row, seed=42):
    """Tries to transform matching questions into Multiple Choice distractors."""
    if row.get('old_exercise_type') != 'matching': return row
    choices = row['choices'] if isinstance(row['choices'], list) else []
    
    pat_upper = r'^[Α-ΩA-Z]\.'       
    pat_lower = r'^([α-ωa-z]|στ)\.'  
    pat_num   = r'^\d+\.'

    col_upper = [c for c in choices if re.match(pat_upper, str(c).strip())]
    col_lower = [c for c in choices if re.match(pat_lower, str(c).strip())]
    col_num   = [c for c in choices if re.match(pat_num, str(c).strip())]
    
    def get_sort_key(txt):
        clean = str(txt).strip()
        m_num = re.match(r'^(\d+)', clean)
        if m_num: return int(m_num.group(1))
        return clean

    col_upper.sort(key=get_sort_key)
    col_lower.sort(key=get_sort_key)
    col_num.sort(key=get_sort_key)

    active_lists = []
    if col_upper: active_lists.append(col_upper)
    if col_lower: active_lists.append(col_lower)
    if col_num:   active_lists.append(col_num)

    if len(active_lists) != 2: return row 

    list_1 = active_lists[0] 
    list_2 = active_lists[1] 

    new_text = str(row['question']) + "\n\n"
    new_text += "Στήλη Α:\n" + "\n".join([str(x) for x in list_1]) + "\n\n"
    new_text += "Στήλη Β:\n" + "\n".join([str(x) for x in list_2])

    correct_pairs_list = row['answer_text'] 
    if not isinstance(correct_pairs_list, list) or not correct_pairs_list: return row

    parsed_pairs = []
    for pair in correct_pairs_list:
        if '-' in pair: parts = pair.split('-', 1)
        elif ' ' in pair: parts = pair.split(' ', 1)
        else: continue 
        parsed_pairs.append((parts[0].strip(), parts[1].strip()))

    if not parsed_pairs: return row

    original_keys = [p[0] for p in parsed_pairs]
    original_vals = [p[1] for p in parsed_pairs]
    rng = random.Random(seed)
    correct_str = ", ".join([f"{k}-{v}" for k, v in zip(original_keys, original_vals)])
    generated_choices = [correct_str]
    
    attempts = 0
    while len(generated_choices) < 4 and attempts < 200:
        shuffled_vals = original_vals[:]
        rng.shuffle(shuffled_vals)
        fake_str = ", ".join([f"{k}-{v}" for k, v in zip(original_keys, shuffled_vals)])
        if fake_str != correct_str and fake_str not in generated_choices:
            generated_choices.append(fake_str)
        attempts += 1

    final_options = generated_choices[:]
    rng.shuffle(final_options)
    try: correct_index = final_options.index(correct_str)
    except ValueError: correct_index = 0

    row['question'] = new_text
    row['choices'] = final_options
    row['answer_text'] = correct_index
    row['answer_index'] = correct_index
    return row

def transform_matching_to_text(row):
    """
    Standardizes matching questions by moving choices into the question text
    and converting the answer list into a clean string.
    """
    choices = row.get('choices', [])
    if not choices:
        return row

    # 1. Format choices as a clean list and append to the question text
    choices_str = "\n".join([str(c) for c in choices])
    row['question'] = f"{row['question']}\n\n{choices_str}"

    # 2. Clear choices to signal it is no longer a fixed-option selection task
    row['choices'] = []

    # 3. Convert the answer to a clean, comma-separated string
    ans = row.get('answer_text')
    if isinstance(ans, str) and ans.strip().startswith('['):
        try:
            parsed = ast.literal_eval(ans)
            if isinstance(parsed, list):
                ans = parsed
        except:
            pass
            
    if isinstance(ans, list):
        row['answer_text'] = ", ".join([str(x) for x in ans])
    elif ans is None:
        row['answer_text'] = ""
    else:
        row['answer_text'] = str(ans)

    # 4. Set answer_index to None to indicate it requires text-based evaluation
    row['answer_index'] = None
    
    return row

def transform_matching_question(row):
    """
    Orchestrator: Tries to transform matching to MC first.
    Falls back to text-based flattening if MC transformation is not possible.
    """
    # 1. Try MC transformation
    row = transform_matching_to_mc(row)
    
    # 2. Check if it's still in the 'complex' state (choices are still a list of more than 4 items, 
    # or it didn't get an answer_index)
    # The MC transformation sets choices to exactly 4 items and answer_index to an int.
    if isinstance(row.get('choices'), list) and len(row.get('choices')) > 4:
        row = transform_matching_to_text(row)
    elif row.get('answer_index') is None and row.get('old_exercise_type') == 'matching':
        # Handles cases where it wasn't a standard 2-column match
        row = transform_matching_to_text(row)
        
    return row

def apply_structural_tags(row):
    """Adds format and reference based on row essence."""
    # 1. format
    et = row.get('old_exercise_type')
    ft_map = {
        'multiple choice': 'multiple_choice',
        'true/false': 'true_false',
        'matching': 'matching',
        'fill-in-the-gaps': 'fill_in_the_gaps',
        'open': 'open_ended'
    }
    row['format'] = ft_map.get(et, 'open_ended')

    # 2. reference
    image_urls = row.get('image_urls')
    subj = str(row.get('subject', '')).lower()
    inp = str(row.get('input', ''))
    
    if image_urls:
        row['reference'] = 'multimodal'
    elif any(kw in inp for kw in ['Πίνακας', 'Πρόγραμμα', 'Πίνακα']):
        row['reference'] = 'table'
    elif (subj in ['greek_language', 'religious studies', 'γλωσσα', 'θρησκευτικα']) and len(inp) > 200:
        row['reference'] = 'passage'
    else:
        row['reference'] = 'none'

    return row

# --- CORE LOGIC ---

def consolidate(subset=None, extended=False):
    """Stages A, B, and C: Data consolidation and processing."""
    print(f"🚀 Starting consolidation {'[Extended]' if extended else ''}... {'[Subset: ' + subset + ']' if subset else '[Full Dataset]'}")
    all_data = []
    data_root = Path("data")
    
    # Walk through data directory
    for root, dirs, files in os.walk(data_root):
        # Normalize paths for platform consistency
        norm_root = os.path.normpath(root)
        if subset and os.path.normpath(subset) not in norm_root:
            continue
            
        json_files = [f for f in files if f.startswith("ΘΕΜΑΤΑ_") and f.endswith(".json")]
        for jf in json_files:
            subject, level, year, exam_set = parse_filename_metadata(jf)
            if not year: continue
            
            json_path = os.path.join(root, jf)
            md_path = os.path.join(root, jf.replace("ΘΕΜΑΤΑ_", "ΑΠΑΝΤΗΣΕΙΣ_").replace(".json", ".md"))
            
            with open(json_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
            
            answers_pool = parse_markdown_answers(md_path)
            ans_tracker = {}
            q_id_counter = {} # Added for unique ID suffixing
            
            for item in questions:
                q_id = str(item.get("id"))
                
                # Handle unique_id suffixing immediately
                if q_id in q_id_counter:
                    q_id_counter[q_id] += 1
                    suffix = f"_{q_id_counter[q_id]}"
                else:
                    q_id_counter[q_id] = 1
                    suffix = ""
                
                # Construct clean ID components
                subj_id_map = {"ΓΛΩΣΣΑ": "greek_language", "ΜΑΘΗΜΑΤΙΚΑ": "math", "ΘΡΗΣΚΕΥΤΙΚΑ": "relig", "ΦΥΣΙΚΗ": "phys"}
                lvl_id_map = {"ΓΥΜΝΑΣΙΟ": "gym", "ΛΥΚΕΙΟ": "lyc"}
                
                subj_part = subj_id_map.get(subject, subject.lower())
                lvl_part = lvl_id_map.get(level, level.lower())
                
                unique_id = f"{subj_part}_{lvl_part}_{year}_{exam_set}_{q_id}{suffix}"

                ans_list = answers_pool.get(q_id, [])
                idx = ans_tracker.get(q_id, 0)
                answer = ans_list[idx] if idx < len(ans_list) else None
                ans_tracker[q_id] = idx + 1
                
                q_text = item.get("question", "")
                q_choices = item.get("choices", [])
                
                # Image processing
                is_multimodal = "no"
                img_basenames = []
                img_full_paths = []
                desc_list = []
                trans_list = []
                for img_obj in item.get("images", []):
                    is_multimodal = "yes"
                    rel_path = img_obj.get("path", "")
                    img_abs = os.path.normpath(os.path.join(root, rel_path))
                    img_basenames.append(os.path.basename(img_abs))
                    img_full_paths.append(img_abs)
                    txt_path = os.path.splitext(img_abs)[0] + ".txt"
                    desc, trans = parse_image_txt(txt_path)
                    if desc: desc_list.append(desc)
                    if trans: trans_list.append(trans)

                ex_type = detect_exercise_type(q_text, q_choices)
                
                raw_mark = ", ".join(item.get("mark", [])) if isinstance(item.get("mark"), list) else item.get("mark")
                
                row = {
                    "id": unique_id,
                    "subject": subject,
                    "admission_level": level,
                    "year": str(year),
                    "exam_set": exam_set,
                    "q_id": q_id,
                    "question": q_text,
                    "input": item.get("input", ""),
                    "choices": q_choices,
                    "answer_text": answer,
                    "answer_index": find_answer_index(q_choices, answer) if ex_type != 'matching' else None,
                    "multimodality": is_multimodal,
                    "image_urls": img_basenames if img_basenames else None,
                    "image_paths": img_full_paths if img_full_paths else [],
                    "image_description": " | ".join(desc_list) if desc_list else None,
                    "image_transcription": " | ".join(trans_list) if trans_list else None,
                    "points": extract_points(raw_mark),
                    "question_type": "closed" if ex_type != 'open' else 'open',
                    "old_exercise_type": ex_type
                }
                
                if ex_type == 'matching':
                    row = transform_matching_question(row)
                
                if extended:
                    row = apply_structural_tags(row)
                    
                all_data.append(row)

    if not all_data:
        print("⚠️ No data found to consolidate.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    
    subject_map = {"ΓΛΩΣΣΑ": "greek_language", "ΜΑΘΗΜΑΤΙΚΑ": "mathematics", "ΘΡΗΣΚΕΥΤΙΚΑ": "religious studies", "ΦΥΣΙΚΗ": "physics"}
    level_map = {"ΓΥΜΝΑΣΙΟ": "gymnasium", "ΛΥΚΕΙΟ": "lyceum"}
    df['subject'] = df['subject'].replace(subject_map)
    df['admission_level'] = df['admission_level'].replace(level_map)
    
    # Organize columns into a narrative flow
    fixed_cols = ['id', 'subject']
    if extended:
        fixed_cols += ['format', 'reference']
    
    fixed_cols += ['question', 'input', 'image_paths', 'image_urls', 'choices', 'answer_text', 'answer_index', 'image_description', 'image_transcription', 'points', 'year', 'admission_level', 'exam_set', 'q_id']
    
    # Filter out columns that might not exist yet (e.g. if extended is False)
    fixed_cols = [c for c in fixed_cols if c in df.columns]
    df = df[fixed_cols + [c for c in df.columns if c not in fixed_cols]]
    
    # Hide internal/legacy columns from the exported dataset
    cols_to_hide = ['multimodality', 'question_type', 'old_exercise_type']
    df = df.drop(columns=[c for c in cols_to_hide if c in df.columns])
    
    # Ensure image lists are stringified consistently for Excel comparison if needed
    # but for internal DF we keep them as lists.
    
    print(f"✅ Consolidation complete. Count: {len(df)}")
    return df

def compare(current_df, reference_file):
    """Compares the consolidated data with a reference Excel file using a merge."""
    if current_df.empty:
        print("❌ Cannot compare: Current dataset is empty.")
        return
    
    print(f"🔍 Comparing with {reference_file}...")
    ref_df = pd.read_excel(reference_file)
    
    # Merge on id
    merged = pd.merge(
        current_df, 
        ref_df, 
        on='id', 
        suffixes=('_cur', '_ref'), 
        how='outer', 
        indicator=True
    )
    
    only_cur = merged[merged['_merge'] == 'left_only']
    only_ref = merged[merged['_merge'] == 'right_only']
    both = merged[merged['_merge'] == 'both']
    
    print(f"📊 Statistics:")
    print(f"   - Match: {len(both)}")
    print(f"   - Only in Current: {len(only_cur)}")
    print(f"   - Only in Reference: {len(only_ref)}")
    
    if not only_cur.empty:
        print(f"⚠️ IDs in current but missing in reference (first 5): {only_cur['id'].head().tolist()}")
    if not only_ref.empty:
        print(f"⚠️ IDs in reference but missing in current (first 5): {only_ref['id'].head().tolist()}")

    # Compare values for rows that exist in both
    cols_to_check = ['subject', 'school_level', 'year', 'answer_text', 'exercise_type']
    for col in cols_to_check:
        col_cur = f"{col}_cur"
        col_ref = f"{col}_ref"
        
        if col_ref not in merged.columns: 
            print(f"❓ Skipping {col}: not in reference.")
            continue
            
        mismatch = (both[col_cur].astype(str).str.strip() != both[col_ref].astype(str).str.strip())
        if mismatch.any():
            print(f"❌ Mismatch in column '{col}': {mismatch.sum()} differences.")
            print(both[mismatch][['id', col_cur, col_ref]].head())
        else:
            print(f"✅ Column '{col}' matches perfectly.")

def push_to_hub(df, with_images=False):
    """Stage D: Push the processed dataset to Hugging Face."""
    repo_id = os.getenv("HF_REPO_ID")
    token = os.getenv("HF_TOKEN")
    is_private = os.getenv("HF_PRIVATE_REPO", "True").lower() == "true"
    gated_setting = os.getenv("HF_GATED_REPO", "False").lower() # False, True, or 'manual'
    
    if not repo_id:
        print("❌ Error: HF_REPO_ID not found in .env")
        return
    
    if not token:
        print("⚠️ Warning: HF_TOKEN not found in .env. Pushing to a private or restricted repo might fail.")
    
    print(f"📤 Preparing to push to Hugging Face Hub: {repo_id} (Private: {is_private}, Gated: {gated_setting})")
    
    try:
        if not repo_exists(repo_id=repo_id, token=token, repo_type="dataset"):
            print(f"🔨 Repository does not exist. Creating {repo_id}...")
            create_repo(repo_id=repo_id, token=token, private=is_private, repo_type="dataset")
            print(f"✅ Created repository: {repo_id}")
            
        # Handle Gating
        if gated_setting in ["true", "manual"]:
            print(f"🔒 Setting repository gating to: {gated_setting}...")
            api = HfApi()
            # If 'true' in .env, we map to True (auto-accept). If 'manual', we keep 'manual'.
            val = True if gated_setting == "true" else "manual"
            api.update_repo_settings(repo_id=repo_id, gated=val, token=token, repo_type="dataset")
            print(f"✅ Gating applied.")

    except Exception as e:
        print(f"⚠️ Error during repo setup/gating: {e}")

    print(f"📊 Pushing {len(df)} rows...")
    
    # Force string type for year and series to prevent numeric formatting on HF
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str)
    if 'exam_set' in df.columns:
        df['exam_set'] = df['exam_set'].astype(str)
    if 'q_id' in df.columns:
        df['q_id'] = df['q_id'].astype(str)
        
    dataset = Dataset.from_pandas(df)
    
    # Hide filename pointers from the final HF benchmark
    if 'image_urls' in dataset.column_names:
        dataset = dataset.remove_columns(["image_urls"])
    
    if with_images and 'image_paths' in df.columns:
        print("🖼️ Casting image_paths to Image features...")
        dataset = dataset.cast_column("image_paths", Sequence(Image()))
        # Rename to the standard 'images' for the Hub
        dataset = dataset.rename_column("image_paths", "images")
    elif 'image_paths' in df.columns:
        # If not pushing images, drop the absolute path column for privacy/cleanliness
        dataset = dataset.remove_columns(["image_paths"])

    try:
        dataset.push_to_hub(repo_id, token=token, private=is_private, split="test")
        print(f"✅ Successfully pushed to Hub (as 'test' split).")
    except Exception as e:
        print(f"❌ Failed to push to Hub: {e}")

# --- CLI ENTRYPOINT ---

def main():
    parser = argparse.ArgumentParser(description="Protipa Exams Dataset Management Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Consolidate command
    con_parser = subparsers.add_parser("consolidate", help="Stages A, B, and C: Consolidate data")
    con_parser.add_argument("--subset", type=str, help="Subset path (e.g., '2025/ΓΥΜΝΑΣΙΟ')")
    con_parser.add_argument("--output", type=str, default="protipa_exams_dataset.xlsx", help="Output filename")

    # Consolidate New command
    con_new_parser = subparsers.add_parser("consolidate_new", help="Stages A-C with Structural Tagging")
    con_new_parser.add_argument("--subset", type=str, help="Subset path")
    con_new_parser.add_argument("--output", type=str, default="protipa_exams_dataset_new.xlsx", help="Output filename")

    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Compare current data with reference")
    comp_parser.add_argument("--reference", type=str, required=True, help="Reference Excel file")
    comp_parser.add_argument("--subset", type=str, help="Subset path to limit comparison")

    # Push command
    push_parser = subparsers.add_parser("push", help="Stage D: Push to Hugging Face Hub")
    push_parser.add_argument("--file", type=str, help="Existing Excel file to push (optional)")
    push_parser.add_argument("--subset", type=str, help="Subset path to push (ignored if --file is used)")
    push_parser.add_argument("--extended", action="store_true", help="Use structural schema and preserve list types")
    push_parser.add_argument("--with-images", action="store_true", help="Embed actual image data (pixels) into the push")

    args = parser.parse_args()

    if args.command == "consolidate":
        df = consolidate(args.subset)
        if not df.empty:
            # Clean illegal characters for Excel
            def clean_illegal(val):
                if isinstance(val, str):
                    return "".join(c for c in val if c.isprintable() or c in "\n\r\t")
                return val
            df = df.map(clean_illegal)
            # Drop absolute paths for Excel export
            if 'image_paths' in df.columns:
                df = df.drop(columns=['image_paths'])
            
            with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Dataset')
                worksheet = writer.sheets['Dataset']
                worksheet.auto_filter.ref = worksheet.dimensions
            print(f"💾 Saved to {args.output}")

    elif args.command == "consolidate_new":
        df = consolidate(args.subset, extended=True)
        if not df.empty:
            # Clean illegal characters for Excel
            def clean_illegal(val):
                if isinstance(val, str):
                    return "".join(c for c in val if c.isprintable() or c in "\n\r\t")
                return val
            df = df.map(clean_illegal)
            # Drop absolute paths for Excel export
            if 'image_paths' in df.columns:
                df = df.drop(columns=['image_paths'])
            
            with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Dataset')
                worksheet = writer.sheets['Dataset']
                worksheet.auto_filter.ref = worksheet.dimensions
            print(f"💾 Saved to {args.output}")

    elif args.command == "compare":
        # Consolidate locally first for comparison
        df = consolidate(args.subset)
        compare(df, args.reference)

    elif args.command == "push":
        if args.file:
            df = pd.read_excel(args.file)
            # If we load from Excel, we must manually safly-eval list strings back to lists
            for col in ['choices', 'image_urls']:
                if col in df.columns:
                    def safe_eval(val):
                        try: 
                            if isinstance(val, str) and val.startswith('['): return ast.literal_eval(val)
                            return val
                        except: return val
                    df[col] = df[col].apply(safe_eval)
        else:
            df = consolidate(subset=args.subset, extended=args.extended)
        
        if not df.empty:
            push_to_hub(df, with_images=args.with_images)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
