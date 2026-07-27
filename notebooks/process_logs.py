# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Process Benchmark Logs & Generate Public/Private Result Tables for Panellinies & Protipa

# %%
import os
import sys
from pathlib import Path
import pandas as pd

_CURRENT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

import process_log_utils as utils  # pyrefly: ignore[missing-import] # type: ignore

# %%
LOGS_DIR = Path('/home/shared/eacl_2026/logs/tmp')

print("Searching for zero-shot log files under /home/shared/eacl_2026/logs/tmp/...")
open_eval_files = [p for p in LOGS_DIR.glob('**/*.eval') if 'zero-shot' in str(p)]
closed_json_files = [p for p in LOGS_DIR.glob('**/*.json') if 'zero-shot' in str(p)]

print(f"Found {len(open_eval_files)} zero-shot open-ended eval files.")
print(f"Found {len(closed_json_files)} zero-shot closed/structured json files.")

# Parse open-ended logs
open_dfs = []
for f in open_eval_files:
    df = utils.parse_open_ended_eval(f)
    if not df.empty:
        df['file_path'] = str(f)
        df['dataset'] = 'protipa' if 'protipa' in str(f) else 'panellinies'
        open_dfs.append(df)

if open_dfs:
    df_all_open = pd.concat(open_dfs, ignore_index=True)
    print(f"Total open-ended samples loaded across files: {len(df_all_open)}")
else:
    df_all_open = pd.DataFrame()

# Parse closed/structured logs
closed_dfs = []
for f in closed_json_files:
    df = utils.parse_closed_structured_json(f)
    if not df.empty:
        df['file_path'] = str(f)
        df['dataset'] = 'protipa' if 'protipa' in str(f) else 'panellinies'
        closed_dfs.append(df)

if closed_dfs:
    df_all_closed = pd.concat(closed_dfs, ignore_index=True)
    print(f"Total closed/structured samples loaded across files: {len(df_all_closed)}")
else:
    df_all_closed = pd.DataFrame()

# %%
def _get_model_key(file_path_str: str) -> str | None:
    fp = file_path_str.lower()
    if 'llama-krikri 8b' in fp or 'krikri-8b' in fp or 'krikri_8b' in fp:
        return 'KriKri-8B'
    elif 'meta-llama 8b' in fp or 'llama3-1-8b' in fp or 'llama3_1_8b' in fp:
        return 'Llama-8B'
    elif 'gemma 4 26b' in fp or 'gemma-4-26b' in fp or 'gemma_4_26b' in fp:
        return 'Gemma-26B'
    elif 'qwen 3 32b' in fp or 'qwen3-32b' in fp or 'qwen_32b' in fp:
        return 'Qwen-32B'
    return None

def build_tables_for_dataset(dataset_name: str) -> dict:
    splits = ['Full', 'Public', 'Private']
    eval_types = ['Closed', 'Struct', 'Open-ended']
    results_table = {s: {m: {et: {} for et in eval_types} for m in utils.MODELS} for s in splits}

    for m_key in utils.MODELS:
        # 1. Closed & Structured
        if not df_all_closed.empty:
            df_ds = df_all_closed[df_all_closed['dataset'] == dataset_name]
            df_m = df_ds[df_ds['file_path'].apply(lambda fp: _get_model_key(fp) == m_key)]
            if not df_m.empty:
                for et in ['Closed', 'Struct']:
                    df_et = df_m[df_m['eval_type'] == et]
                    if not df_et.empty:
                        # Full
                        sub_acc_full = df_et.groupby('subject')['score'].mean() * 100.0
                        for subj, acc in sub_acc_full.items():
                            if subj in utils.SUBJECT_ORDER:
                                results_table['Full'][m_key][et][subj] = acc
                        results_table['Full'][m_key][et]['Aggregate'] = df_et['score'].mean() * 100.0

                        # Public (< 2026)
                        df_pub = df_et[~df_et['is_private']]
                        if not df_pub.empty:
                            sub_acc_pub = df_pub.groupby('subject')['score'].mean() * 100.0
                            for subj, acc in sub_acc_pub.items():
                                if subj in utils.SUBJECT_ORDER:
                                    results_table['Public'][m_key][et][subj] = acc
                            results_table['Public'][m_key][et]['Aggregate'] = df_pub['score'].mean() * 100.0

                        # Private (== 2026)
                        df_priv = df_et[df_et['is_private']]
                        if not df_priv.empty:
                            sub_acc_priv = df_priv.groupby('subject')['score'].mean() * 100.0
                            for subj, acc in sub_acc_priv.items():
                                if subj in utils.SUBJECT_ORDER:
                                    results_table['Private'][m_key][et][subj] = acc
                            results_table['Private'][m_key][et]['Aggregate'] = df_priv['score'].mean() * 100.0

        # 2. Open-ended
        if not df_all_open.empty:
            df_ds = df_all_open[df_all_open['dataset'] == dataset_name]
            df_m = df_ds[df_ds['file_path'].apply(lambda fp: _get_model_key(fp) == m_key)]
            if not df_m.empty:
                # Full
                sub_acc_full = df_m.groupby('subject')['score'].mean() * 100.0
                for subj, acc in sub_acc_full.items():
                    if subj in utils.SUBJECT_ORDER:
                        results_table['Full'][m_key]['Open-ended'][subj] = acc
                results_table['Full'][m_key]['Open-ended']['Aggregate'] = df_m['score'].mean() * 100.0

                # Public (< 2026)
                df_pub = df_m[~df_m['is_private']]
                if not df_pub.empty:
                    sub_acc_pub = df_pub.groupby('subject')['score'].mean() * 100.0
                    for subj, acc in sub_acc_pub.items():
                        if subj in utils.SUBJECT_ORDER:
                            results_table['Public'][m_key]['Open-ended'][subj] = acc
                    results_table['Public'][m_key]['Open-ended']['Aggregate'] = df_pub['score'].mean() * 100.0

                # Private (== 2026)
                df_priv = df_m[df_m['is_private']]
                if not df_priv.empty:
                    sub_acc_priv = df_priv.groupby('subject')['score'].mean() * 100.0
                    for subj, acc in sub_acc_priv.items():
                        if subj in utils.SUBJECT_ORDER:
                            results_table['Private'][m_key]['Open-ended'][subj] = acc
                    results_table['Private'][m_key]['Open-ended']['Aggregate'] = df_priv['score'].mean() * 100.0

    return results_table

# %%
panellinies_tables = build_tables_for_dataset('panellinies')
protipa_tables = build_tables_for_dataset('protipa')

output_md_path = Path(__file__).parent / 'benchmark_results_tables.md'
with open(output_md_path, 'w', encoding='utf-8') as f:
    f.write("# Benchmark Results Tables (Panellinies & Protipa)\n\n")

    f.write("## 1. Panellinies Benchmark Results Tables\n\n")
    for s_name in ['Full', 'Public', 'Private']:
        f.write(f"### Panellinies - {s_name} Benchmark Results Table\n\n")
        f.write("```latex\n")
        f.write(utils.generate_latex_table(panellinies_tables[s_name], f"Pan-Ex ({s_name} Benchmark)"))
        f.write("\n```\n\n")

    f.write("## 2. Protipa Benchmark Results Tables\n\n")
    for s_name in ['Full', 'Public', 'Private']:
        f.write(f"### Protipa - {s_name} Benchmark Results Table\n\n")
        f.write("```latex\n")
        f.write(utils.generate_latex_table(protipa_tables[s_name], f"Prot-Ex ({s_name} Benchmark)"))
        f.write("\n```\n\n")

print(f"\nGenerated Markdown and LaTeX tables saved to {output_md_path}")
