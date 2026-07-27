import zipfile
import json
import tarfile
from pathlib import Path

print("=== Examining all files in closed_structured_logs ===")
closed_dir = Path('/home/shared/eacl_2026/logs/tmp/closed_structured_logs')
for p in closed_dir.glob('*'):
    print(p.name, p.stat().st_size)
    if p.suffix == '.zip':
        with zipfile.ZipFile(p) as z:
            print("  Zip files:", z.namelist())

print("\n=== Examining all files in open_ended_logs ===")
open_dir = Path('/home/shared/eacl_2026/logs/tmp/open_ended_logs')
for p in open_dir.glob('*'):
    print(p.name, p.stat().st_size)
    # Check if .eval is a zip archive (Inspect AI eval logs are zstandard zip files)
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            print("  Eval zip files:", z.namelist()[:5])
    else:
        print("  Not standard zipfile format")
