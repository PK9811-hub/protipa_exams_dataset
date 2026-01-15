# Protipa Exams Dataset Evaluation

This project evaluates Large Language Models (LLMs) on the Greek Protipa Exams dataset.

## Project Structure

- `data/`: Contains raw exam data (PDFs, DOCX, JSON, MD).
- `notebooks/`: Jupyter notebooks for evaluation and analysis.
- `src/protipa_exams_dataset/`: Main source code for data loading and evaluation logic.
- `pyproject.toml`: Project configuration and dependencies.
- `.env`: Environment variables (API keys, host URLs).

## Getting Started

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Configure `.env`:
   ```bash
   LITELLM_HOST=your_host_url
   LITELLM_ILSP_EVAL_API_KEY=your_api_key
   ```
3. Run evaluation notebooks in `notebooks/`.

## Development / Editable Mode

To ensure that the package is installed in **editable mode** (so that changes to the files in `src/` are reflected immediately), use:

```bash
uv pip install -e .
```

When the project is installed in editable mode, you can combine it with **IPython autoreload** in your notebooks to develop source code and run experiments simultaneously:

```python
%load_ext autoreload
%autoreload 2

from protipa_exams_dataset import load_protipa_dataset
# Now any changes to data_loader.py will be automatically reloaded!
```

## Usage

You can use the utility functions in your notebooks by importing them from their respective modules:

```python
from protipa_exams_dataset.data_loader import load_protipa_dataset, filter_dataset, apply_matching_processing, clean_dataset_paths

dataset = load_protipa_dataset()
df = dataset.to_pandas()

# Process matching exercises to create distractors and shuffle
df_final = apply_matching_processing(df)

# Clean paths to keep only filenames
df_final = clean_dataset_paths(df_final)
```
