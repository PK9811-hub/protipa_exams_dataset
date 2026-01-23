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
   
| Key | Description |
|-----|-------------|
| **id** | Structural breadcrumb identifier `{subject}_{level}_{year}_{exam_set}_{q_id}` (e.g., `math_gym_2023_1_5`). |
| **question** | The introductory text and the core task of the exercise, including any necessary formulas or equations encoded in $\text{LaTeX}$. |
| **input** | Supplementary text provided with the exercise, such as literary passages (for Greek Language) or detailed descriptions of diagrams/images. |
| **choices** | The candidate answers provided for closed-ended question types. |
| **images** | A list of objects containing the file path, detailed description, and transcription of any accompanying images or diagrams (multi-modal components). |
| **points** | The assigned score/points for the item (numeric). |


2. **Dataframe Columns (Excel/CSV)**

| Column | Description |
|--------|-------------|
| **id** | Standardized unique identifier for the row entry (breadcrumb format). |
| **subject** | The academic subject (e.g., `greek_language`, `mathematics`, `physics`, `religious studies`). |
| **format** | (New) Interaction format: `multiple_choice`, `true_false`, `matching`, `fill_in_the_gaps`, `open_ended`. |
| **reference** | (New) Contextual requirement/addenda: `passage`, `multimodal`, `table`, `none`. |
| **question** | The introductory text/core task. |
| **input** | Supplementary text (passages, diagram descriptions). |
| **images** | The actual image asset(s) (Hugging Face only). |
| **choices** | Candidate answers for closed-ended questions. |
| **answer_text** | The processed, validated answer/solution (string). |
| **answer_index** | The numeric mapping of the answer (for programmatic evaluation). |
| **image_description** | Textual proxy for visual content. |
| **image_transcription** | OCR/Text extraction from within the visual assets. |
| **image_urls** | Basemate pointers of the source image file(s) (Excel only). |
| **points** | Assigned point value for the question (numeric). |
| **year** | The year of the exam (stored as a string). |
| **admission_level** | The target education level: `gymnasium` or `lyceum`. |
| **exam_set** | The ID of the exam batch/paper (e.g., 1, 2). |
| **q_id** | The specific ID/number of the question within its set. |



3. **Subject Coverage and Question Types**

| File / Directory     | Description |
|----------------------|-------------|
| **Greek Language**   | Covers all defined task types (MC, T/F, Gaps, Matching). |
| **Mathematics**      | Primarily Open-ended tasks and Multiple-Choice. |
| **Physics**          | Exclusively Open-ended questions. |
| **Religious Studies**| Exclusively Closed-ended questions (Multiple-Choice). |




⚠️ **Known Data Gaps and Missing Information**
Due to the nature of the publicly available source files, the following gaps were identified:

• **Missing Points**: The `points` column is only partially populated (~30% of rows). Scoring data is primarily present in older historical papers (2013-2019) and the latest 2025 papers. For the 2020-2024 period, point values are often missing in the source files.

• **Missing Year**: All exam files from 2015 were unavailable, resulting in a gap in the time series data.

• **Missing Solutions**: Official solutions were not provided for:
  - `greek_language` (Lyceum, 2014)
  - `greek_language` and `mathematics` (Lyceum, 2018)

• **Writing Prompts**: For the free-text writing component of the `greek_language` exams, a separate file is provided containing only the prompts and grading guidelines, as official model answers do not exist.


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

# Load the test split (default for benchmarks)
dataset = load_protipa_dataset(split='test') 
df = dataset.to_pandas()

# Process matching exercises to create distractors and shuffle
df_final = apply_matching_processing(df)

# Clean paths to keep only filenames
df_final = clean_dataset_paths(df_final)
```

---

## 🏗️ Schema Evolution & Rationale

This section summarizes the architectural transition from the legacy "Extraction Schema" (found in the original Excel files) to the new "Structural Benchmark Schema."

### **1. Key Property Transitions**

| Feature | Legacy Schema (Old) | Structural Schema (New) | Rationale |
| :--- | :--- | :--- | :--- |
| **Primary Key** | `unique_id` | **`id`** | Standardized breadcrumb format `{subj}_{lvl}_{yr}_{set}_{q_id}` for perfect provenance tracing. |
| **Multi-modal** | `multimodality` | **`reference`** | Transitions from a simple yes/no flag to a specific asset type (e.g., multimodal). |
| **Numeric Data** | `year` (int) | **`year`** (string) | Prevents Excel/HF from displaying years as numbers (e.g., `2,024`). |
| **Level Context** | `school_level` | **`admission_level`** | Clarifies that these are entrance exams *for* the level, not exams taken *at* that level. |
| **Question ID** | `label_id` (float-like) | **`q_id`** (string) | Standardized naming and preserves original formatting (e.g., `1.1` stays `1.1`). |
| **Exam Set** | `series` | **`exam_set`** | Renamed to better reflect that these are distinct batches/papers of the same exam year. |
| **Subject Naming** | `modern greek` | **`greek_language`** | Aligns with standard academic NLP subject naming conventions. |
| **Split Naming** | `train` | **`test`** | Signals that the data is an official assessment suite (benchmark), not for training. |
| **Access Control** | None (Public) | **Gated Access** | Implements manual/auto approval via Hugging Face to protect data integrity. |

### **2. The "Structural Duo" (New Columns)**

We introduced two new structural columns to shift the dataset from a "data dump" to a **diagnostic benchmark**. These columns are derived using **Expert-Rule Heuristics** (regex + length + domain mapping).

#### **A. `format`** (The Interaction Model)
*   **From**: `exercise_type` (e.g., `multiple-choice`).
*   **To**: `format` (standardized: `matching`, `fill_in_the_gaps`).
*   **Rationale**: Defines *how* the model must interact with the answer space.
    *   *Note on Matching*: While standard 2-column matching questions are ideally transformed into Multiple Choice distractors for easier evaluation, the 4 items currently in this dataset failed this transformation due to their complex (n-to-m) structure. These were resolved by moving the matchable items directly into the `question` field and providing the answer as a stringified list of pairs.

#### **B. `reference`** (The Data Requirement)
*   **Origin**: Newly generated via Heuristics.
*   **Values**: `passage`, `multimodal`, `table`, `none`.
*   **Rationale**: Identifies **which additional asset type** (addenda) the model must parse to succeed. This allows for fine-grained measurement of failure points (e.g., multimodal errors vs. reading comprehension errors).

### **3. Why these Heuristics are used?**
Instead of black-box AI tagging, we use **deterministic heuristics** (e.g., `len(input) > 200` triggers `passage`).
*   **Consistency**: Every run generates the same metadata.
*   **Transparency**: No "hidden" AI tagging; rules are based on the raw data essence.
*   **Diagnostic Power**: Enables pinpointing if a model is "Reading-Heavy" vs "Logic-Heavy."
