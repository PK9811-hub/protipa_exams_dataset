import zipfile
import json
import zstandard as zstd
import io

class ZstdZipFile(zipfile.ZipFile):
    def read(self, name, pwd=None):
        info = self.getinfo(name)
        if info.compress_type == 93:
            with self.open(name, pwd=pwd, force_zip64=True) as f:
                # Bypass built-in decompressor check by reading raw bytes from ZipExtFile
                pass
        return super().read(name, pwd=pwd)

def read_inspect_eval(eval_path):
    with open(eval_path, 'rb') as f:
        zbuf = io.BytesIO(f.read())
    
    zf = zipfile.ZipFile(zbuf)
    sample_files = [fn for fn in zf.namelist() if fn.startswith('samples/')]
    print(f"Total samples in inspect eval file: {len(sample_files)}")
    
    # Check sample files list for years
    years = set()
    sample_metadata_years = set()
    
    for fn in sample_files:
        # File name example: samples/ancient_greek_gel_2026_Α1.β.ii_epoch_1.json
        parts = Path(fn).stem.split('_')
        for p in parts:
            if p.isdigit() and len(p) == 4:
                years.add(int(p))
                
    print("Years found in sample filenames:", sorted(list(years)))

def read_lmeval_results(zip_path):
    print("\n=== Closed-Structured (lm-eval) Zip ===")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith('.json'):
                data = json.loads(z.read(name))
                results = data.get('results', {})
                print("Tasks in results:", list(results.keys()))

if __name__ == '__main__':
    from pathlib import Path
    open_file = '/home/shared/eacl_2026/logs/tmp/open_ended_logs/2026-07-24_llama3-1-8b_open_ended_0-shot_t_0.1_top_p_0.9_top_k_40.eval'
    closed_file = '/home/shared/eacl_2026/logs/tmp/closed_structured_logs/2026-07-25_llama3_1_8b_closed_structured_0shot.zip'
    
    read_inspect_eval(open_file)
    read_lmeval_results(closed_file)
