import zipfile
import json
from pathlib import Path

open_file = Path('/home/shared/eacl_2026/logs/tmp/open_ended_logs/2026-07-24_llama3-1-8b_open_ended_0-shot_t_0.1_top_p_0.9_top_k_40.eval')

with zipfile.ZipFile(open_file) as z:
    sample_files = [f for f in z.namelist() if f.startswith('samples/')]
    print(f"Total sample files in eval: {len(sample_files)}")
    print("First 10 sample file names:", sample_files[:10])
    
    # Read one sample json
    sample_content = json.loads(z.read(sample_files[0]))
    print("\nKeys in sample json:", list(sample_content.keys()))
    print("Sample id:", sample_content.get('id'))
    print("Sample metadata:", sample_content.get('metadata'))
