import json
import zipfile_zstd as zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

SUBJECT_MAPPING = {
    'ancient_greek': 'Ancient Greek',
    'economics': 'Economics',
    'physics': 'Physics',
    'history': 'History',
    'latin': 'Latin',
    'mathematics': 'Mathematics',
    'greek_language': 'Greek Language',
    'computer_science': 'Computer Science',
    'chemistry': 'Chemistry',
    'biology': 'Biology'
}

SUBJECT_ORDER = [
    'Ancient Greek',
    'Economics',
    'Physics',
    'History',
    'Latin',
    'Mathematics',
    'Greek Language',
    'Computer Science',
    'Chemistry',
    'Biology'
]

MODELS = ['KriKri-8B', 'Llama-8B', 'Gemma-26B', 'Qwen-32B']


def parse_open_ended_eval(eval_path: str | Path) -> pd.DataFrame:
    eval_path = Path(eval_path)
    samples = []

    with zipfile.ZipFile(eval_path) as z:
        for info in z.infolist():
            if info.filename.startswith('samples/'):
                raw_bytes = z.read(info)
                sample_json = json.loads(raw_bytes)
                sample_id = sample_json.get('id', '')

                metadata = sample_json.get('metadata', {})
                year = metadata.get('year')
                if year is None:
                    parts = str(sample_id).split('_')
                    for p in parts:
                        if p.isdigit() and len(p) == 4:
                            year = int(p)
                            break
                if year is not None:
                    year = int(year)

                subject = metadata.get('subject')
                if not subject:
                    for s_key, s_name in SUBJECT_MAPPING.items():
                        if s_key in str(sample_id).lower():
                            subject = s_name
                            break
                else:
                    subject = SUBJECT_MAPPING.get(str(subject).lower(), subject)

                scores = sample_json.get('scores', {})
                score_val = None
                if scores:
                    for score_k, s_obj in scores.items():
                        if isinstance(s_obj, dict):
                            v = s_obj.get('value')
                            if isinstance(v, (int, float)):
                                score_val = float(v)
                                break
                            elif isinstance(v, bool):
                                score_val = 1.0 if v else 0.0
                                break
                            elif isinstance(v, dict):
                                acc = v.get('accuracy') or v.get('score') or v.get('exact_match')
                                if acc is not None:
                                    score_val = float(acc)
                                    break

                samples.append({
                    'sample_id': sample_id,
                    'year': year,
                    'is_private': (year == 2026),
                    'subject': subject,
                    'score': score_val
                })

    return pd.DataFrame(samples)


def parse_closed_structured_json(json_path: str | Path) -> pd.DataFrame:
    json_path = Path(json_path)
    samples = []
    with open(json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)

    samples_dict = content.get('samples', {})
    for task_name, sample_list in samples_dict.items():
        eval_type = 'Struct' if 'structured' in task_name else 'Closed'

        for s in sample_list:
            doc = s.get('doc', {})
            year = doc.get('year')
            if year is not None:
                year = int(year)

            subj_raw = doc.get('subject') or task_name
            subject = None
            for s_key, s_name in SUBJECT_MAPPING.items():
                if s_key in str(subj_raw):
                    subject = s_name
                    break

            score_val = None
            if 'exact_match' in s:
                score_val = float(s['exact_match'])
            elif 'structured_accuracy' in s:
                score_val = float(s['structured_accuracy'])

            samples.append({
                'task_name': task_name,
                'eval_type': eval_type,
                'subject': subject,
                'year': year,
                'is_private': (year == 2026),
                'score': score_val
            })

    return pd.DataFrame(samples)


def generate_latex_table(results_dict: dict, title: str) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \resizebox{\textwidth}{!}{",
        r"    \begin{tabular}{l ccc ccc ccc ccc}",
        r"        \toprule",
        r"        \multirow{3}{*}{\textbf{Subject}} & \multicolumn{3}{c}{\textbf{KriKri-8B}} & \multicolumn{3}{c}{\textbf{Llama-8B}} & \multicolumn{3}{c}{\textbf{Gemma-26B}} & \multicolumn{3}{c}{\textbf{Qwen-32B}} \\",
        r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}",
        r"        & \multirow{2}{*}{\textbf{Closed}} & \multicolumn{2}{c}{\textbf{Open}} & \multirow{2}{*}{\textbf{Closed}} & \multicolumn{2}{c}{\textbf{Open}} & \multirow{2}{*}{\textbf{Closed}} & \multicolumn{2}{c}{\textbf{Open}} & \multirow{2}{*}{\textbf{Closed}} & \multicolumn{2}{c}{\textbf{Open}} \\",
        r"        \cmidrule(lr){3-4} \cmidrule(lr){6-7} \cmidrule(lr){9-10} \cmidrule(lr){12-13}",
        r"        & & \textit{Struct.} & \textit{Open-ended} & & \textit{Struct.} & \textit{Open-ended} & & \textit{Struct.} & \textit{Open-ended} & & \textit{Struct.} & \textit{Open-ended} \\",
        r"        \midrule"
    ]

    for subj in SUBJECT_ORDER:
        row_cells = [subj]
        for model in MODELS:
            val_c = results_dict.get(model, {}).get('Closed', {}).get(subj)
            str_c = f"{val_c:.1f}" if val_c is not None else "--"

            val_s = results_dict.get(model, {}).get('Struct', {}).get(subj)
            str_s = f"{val_s:.1f}" if val_s is not None else "--"

            val_o = results_dict.get(model, {}).get('Open-ended', {}).get(subj)
            str_o = f"{val_o:.1f}" if val_o is not None else "--"

            row_cells.extend([str_c, str_s, str_o])

        lines.append("        " + " & ".join(row_cells) + r" \\")

    lines.append(r"        \midrule")

    agg_cells = [r"\textbf{Aggregate}"]
    for model in MODELS:
        val_c = results_dict.get(model, {}).get('Closed', {}).get('Aggregate')
        str_c = f"\\textbf{{{val_c:.1f}}}" if val_c is not None else "--"

        val_s = results_dict.get(model, {}).get('Struct', {}).get('Aggregate')
        str_s = f"\\textbf{{{val_s:.1f}}}" if val_s is not None else "--"

        val_o = results_dict.get(model, {}).get('Open-ended', {}).get('Aggregate')
        str_o = f"\\textbf{{{val_o:.1f}}}" if val_o is not None else "--"

        agg_cells.extend([str_c, str_s, str_o])

    lines.append("        " + " & ".join(agg_cells) + r" \\")
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    }",
        r"    \vspace{-0.2cm}",
        f"    \\caption{{Zero-shot performance comparison (%) for {title}.}}",
        f"    \\label{{tab:{title.lower().replace(' ', '_')}_panellinies_results}}",
        r"\end{table*}"
    ])

    return "\n".join(lines)
