📚 **GR-ProtipaExams Dataset: Structured Exam Questions (2013-2025)**

**Overview**


The GR-ProtipaExams project introduces a comprehensive, structured dataset, called “GR-ProtipaExams”, derived from publicly available exam questions and solutions used for student admission to Model and Experimental Schools (Protipa and Peiramatika Schools) in Greece. This dataset spans the years 2013 to 2025 and covers core secondary education subjects.

The primary goal is to provide a standardized resource for educational research, quantitative statistical analysis, and, particularly, for training, finetuning, and evaluating Large Language Models (LLMs) on complex, multi-subject assessment tasks in the Greek language.


**Repository Contents**

This repository is strictly dedicated to hosting the final, processed data artifacts:

• JSON files: Contain the structured format of individual questions and their complete metadata.

• MD files: Markdown versions of the correct answers/solutions, designed for easy viewing and LLM consumption.

• Excel / CSV dataframe: A consolidated, analysis-ready tabular file for filtering, statistical tasks, and quick data exploration.


🔗 **Source Code and Data Generation**

The entire pipeline—from corpus gathering and cleaning to structuring, ID generation, and alignment—was implemented through dedicated scripts.

The full source code repository used to generate this dataset is located here:

[ https://github.com/PK9811-hub/dataset_creation ]



1. **Data Structure and Keys**
   
| File / Directory | Description |
|------------------|-------------|
| **id** | Unique identifier for each question–answer entry (e.g., `MATH_2023_HS_Q05`). |
| **question** | The introductory text and the core task of the exercise, including any necessary formulas or equations encoded in $\text{LaTeX}$. |
| **input** | Supplementary text provided with the exercise, such as literary passages (for Greek Language) or detailed descriptions of diagrams/images. |
| **choices** | The candidate answers provided for closed-ended question types. |
| **images** | A list of objects containing the file path, detailed description, and transcription of any accompanying images or diagrams (multi-modal components). |
| **mark** | The assigned score/mark for each correct entry. |


2. **Dataframe Columns (Excel/CSV)**

| File / Directory | Description |
|------------------|-------------|
| **unique_id** | Unique identifier for the row entry (autoincremented index for the dataframe). |
| **subject** | The academic subject of the exam (e.g., Greek Language, Math, Physics, Religious Studies). |
| **school_level** | The education level (middle school or high school). |
| **series** | Refers to the original question numbering from the JSON file (e.g., 1.1, 1.2, etc.). |
| **label_id** | The original ID of the question grouping as it appeared in the raw exam files (e.g., 1, 2, 3). |
| **question** | The introductory text and the core task of the exercise, including LaTeX. |
| **input** | Supplementary text (passages, diagram descriptions). |
| **choices** | The candidate answers provided for closed-ended questions. |
| **answer** | The final, validated answer/solution. |
| **multimodality** | Indicates the presence of associated diagrams or images (yes/no). |
| **image_path** | Local file path to the image file used in the question. |
| **image_link** | External link/URL associated with the image. |
| **mark** | The assigned score/mark for the entry. |
| **question_type** | The general structure: Open or Closed ended. |
| **exercise_type** | The specific format: multiple-choice, true/false, fill-in-the-gaps, or matching. |
| **source_file** | The name of the original file/document from which the question was extracted. |



3. **Subject Coverage and Question Types**

| File / Directory     | Description |
|----------------------|-------------|
| **Greek Language**   | Covers all defined task types (MC, T/F, Gaps, Matching). |
| **Mathematics**      | Primarily Open-ended tasks and Multiple-Choice. |
| **Physics**          | Exclusively Open-ended questions. |
| **Religious Studies**| Exclusively Closed-ended questions (Multiple-Choice). |




⚠️ **Known Data Gaps and Missing Information**
Due to the nature of the publicly available source files, the following gaps were identified and processed:

• Missing Year: All exam files from 2015 were unavailable, resulting in a gap in the time series data.

• Missing Solutions: Official solutions were not provided for:

Greek Language (High School, 2014)

Greek Language and Mathematics (High School, 2018)

•  Writing Prompts: For the free-text writing component of the Greek Language exams, a separate file is provided containing only the prompts and grading guidelines, as official model answers do not exist.


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
