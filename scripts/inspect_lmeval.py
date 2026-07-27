import zipfile
import json
import os

def inspect_closed_structured(zip_path):
    print("=== Closed Structured (lm-eval) ===")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith('.json'):
                data = json.loads(z.read(name))
                samples_dict = data.get('samples', {})
                for task_name, samples in samples_dict.items():
                    print(f"Task: {task_name}, Sample Count: {len(samples)}")
                    if samples:
                        s0 = samples[0]
                        print("Sample keys:", list(s0.keys()))
                        print("Sample doc keys:", list(s0.get('doc', {}).keys()))
                        print("Sample doc sample:", s0.get('doc'))
                        print("Sample doc_id:", s0.get('doc_id'))

if __name__ == '__main__':
    closed_path = '/home/shared/eacl_2026/logs/tmp/closed_structured_logs/2026-07-25_llama3_1_8b_closed_structured_0shot.zip'
    inspect_closed_structured(closed_path)
