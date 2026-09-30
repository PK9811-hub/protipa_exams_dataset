import os
import zipfile
import json
import io
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

class ZstdZipFile(zipfile.ZipFile):
    def read(self, name, pwd=None):
        info = self.getinfo(name)
        if info.compress_type == 93:
            with self.open(name, pwd=pwd, force_zip64=True) as f:
                pass
        return super().read(name, pwd=pwd)

def is_in_our_subsets(sample_id):
    sid = sample_id.lower()
    if 'gym' in sid and any(y in sid for y in ['2016', '2018']) and any(m in sid for m in ['language', 'math']):
        return True
    return False

def clean_prefix(text):
    if not isinstance(text, str):
        return str(text)
    
    return re.sub(r'^(Question|Context|Answer):\s*', '', text, flags=re.IGNORECASE).strip()

def parse_input_to_dict(text, img_desc):
    if not isinstance(text, str):
        text = str(text)
        
    parsed = {}
    
    text = re.sub(r'\[Image Description:.*?\]\n*', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r'\[Image Transcription:.*?\]\n*', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    
    has_context = "Context:" in text
    has_question = "Question:" in text
    
    if has_context and has_question:
        parts = text.split("Question:")
        parsed["context"] = parts[0].replace("Context:", "").strip()
        parsed["question"] = parts[1].strip()
    elif has_question:
        parsed["question"] = text.replace("Question:", "").strip()
    elif has_context:
        parsed["context"] = text.replace("Context:", "").strip()
    else:
        parsed["question"] = text.strip()
        
    if img_desc:
        parsed["image_description"] = img_desc.strip()
        
    return parsed

def build_dataframe(log_dir):
    log_dir_path = Path(log_dir)
    rows = []
    
    target_models = ['krikri', 'llama', 'gemma', 'qwen']
    
    model_names_mapping = {
        'krikri': 'krikri-8b-instruct-zero-shot',
        'llama': 'llama-3.1-8b-instruct-zero-shot',
        'gemma': 'gemma-4-26b-it-zero-shot',
        'qwen': 'qwen3-32b-zero-shot'
    }

    eval_files = [f for f in log_dir_path.rglob('*.eval') if '0-shot' in f.name.lower()]

    for eval_path in eval_files:
        filename = eval_path.name.lower()
        
        matched_model = next((m for m in target_models if m in filename), None)
        if not matched_model:
            continue

        full_model_name = model_names_mapping[matched_model]
        print(f"Processing: {eval_path.name} (Model: {full_model_name})")

        with open(eval_path, 'rb') as f:
            zbuf = io.BytesIO(f.read())
        
        zf = ZstdZipFile(zbuf)
        sample_files = [fn for fn in zf.namelist() if fn.startswith('samples/')]
        
        for fn in sample_files:
            sample_data = json.loads(zf.read(fn))
            sample_id = sample_data.get('id', '')
            metadata = sample_data.get('metadata', {})
            
            if metadata.get('format') != 'open_ended':
                continue
            
            if is_in_our_subsets(sample_id):
                subject = 'Mathematics' if 'math' in sample_id.lower() else 'Modern Greek'
                scores = sample_data.get('scores', {})
                
                raw_input = sample_data.get('input', '')
                raw_target = sample_data.get('target', '')
                raw_answer = sample_data.get('output', {}).get('completion', '')
                img_desc = metadata.get('image_description')
                
                input_dict = parse_input_to_dict(raw_input, img_desc)
                target_dict = {"reference": clean_prefix(raw_target)}
                answer_dict = {"answer": clean_prefix(raw_answer)}
                
                rows.append({
                    'Subject': subject,
                    'Model': full_model_name,
                    'Question_ID': sample_id,
                    'Input_Question': json.dumps(input_dict, ensure_ascii=False),
                    'Reference_Target': json.dumps(target_dict, ensure_ascii=False),
                    'Model_Answer': json.dumps(answer_dict, ensure_ascii=False),
                    'LLM_Judge_Score': scores.get('generic_judge_scorer', {}).get('value', None),
                    'BERTScore': scores.get('greek_bertscore', {}).get('value', None)
                })
                
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.drop_duplicates(subset=['Model', 'Question_ID'], keep='last')
    
    return df

if __name__ == '__main__':
    load_dotenv()
    open_ended_dir = os.getenv('PROTIPA_LOGS_DIR')
    
    if not open_ended_dir:
        raise ValueError("The PROTIPA_LOGS_DIR is not found in the .env file!")
    
    print("Starting the parsing process")
    final_df = build_dataframe(open_ended_dir)
    
    print(f"\nTotal records found: {len(final_df)}")
    if not final_df.empty:
        final_df.to_csv('human_evaluation_protipa_subset.csv', index=False, encoding='utf-8-sig')
        print("The file human_evaluation_protipa_subset.csv was created successfully!")