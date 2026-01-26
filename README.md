---
configs:
- config_name: default
  data_files:
  - split: test
    path: data/*.parquet
license: cc-by-nc-4.0
task_categories:
- question-answering
language:
- el
tags:
- multimodal
- education
- exams
- greek
size_categories:
- 1K<n<10K
---

**GR-ProtipaExams Dataset: Structured Exam Questions (2013-2025)**

**Overview**

The GR-ProtipaExams project introduces a comprehensive, structured dataset, called “GR-ProtipaExams”, derived from publicly available exam questions and solutions used for student admission to Model and Experimental Schools (Πρότυπα και Πειραματικά Σχολεία) in Greece. Spanning from 2013 to 2025, the dataset provides a rich diversity of:

*   **Subjects**: Core curriculum coverage including Greek Language, Mathematics, Physics, and Religious Studies.
*   **Educational Levels**: Questions targeted at both Gymnasium and Lyceum entrance requirements.
*   **Interaction Formats**: A variety of task types including Multiple Choice, True/False, Matching, Fill-in-the-Gaps, and Open-Ended questions.
*   **Modalities**: Native support for **multimodal evaluation**, featuring high-fidelity images/diagrams, textual transcriptions, and structural references (tables and passages).

The primary goal of this repository is to establish a rigorous, standardized benchmark for the **evaluation of Large Language Models (LLMs)** on complex, multi-subject assessment tasks in the Greek language. Additionally, it serves as a high-quality resource for educational research and quantitative statistical analysis.

 The source material was obtained from the official portal of the **Governing Body of Model and Experimental Schools** ([https://depps.minedu.gov.gr/](https://depps.minedu.gov.gr/)).

> [!IMPORTANT]
> **Disclaimer**: While every effort has been made to ensure the accuracy and completeness of this structured dataset, any errors, omissions, or formatting issues are the result of the processing and transformation pipeline and are **not related** to the original source or the Ministry of Education.

---

The dataset is available on Hugging Face at [https://huggingface.co/ilsp/GR-ProtipaExams](https://huggingface.co/ilsp/GR-ProtipaExams).

If you make use of this dataset, please consider citing it as follows:

```bibtex
@misc{ilsp-gr-protipa-exams,
    author = {Kyriazi, Penny and Prokopidis, Prokopis},
    title = {GR-ProtipaExams},
    howpublished = {\url{https://huggingface.co/ilsp/GR-ProtipaExams}},
    year = {2025}
}
```

##  **Dataset Columns**

| Column | Description |
|--------|-------------|
| **id** | Unique identifier. |
| **subject** | Academic subject (including `greek_language`, `mathematics`, `physics`, `religious studies`). |
| **format** | `multiple_choice`, `true_false`, `matching`, `fill_in_the_gaps`, `open_ended`. |
| **reference** | Additional reference inputs: (text) `passage`, `multimodal`, `table`, `none`. |
| **question** | The core question. |
| **input** | Supplementary text (passages, diagram and image presentations). |
| **images** | **(Visual)** The actual image asset(s) rendered as pixel data. |
| **choices** | Candidate answers for closed-ended questions. |
| **answer_text** | The processed, validated answer/solution (string). |
| **answer_index** | The numeric mapping of the answer for closed-ended questions. |
| **image_description** | Detailed textual proxy for visual content. |
| **image_transcription** | OCR/Text extraction from within the visual assets. |
| **points** | Assigned point value for the question (numeric). May be missing in the source. |
| **year** | The year of the exam. |
| **admission_level** | The target education level: `gymnasium` or `lyceum`. |
| **exam_set** | The ID of the exam batch/paper (e.g., 1, 2) |
| **q_id** | The specific ID/number of the question within its set. |

### **Usage (Hugging Face)**

```python
from datasets import load_dataset

# Load the benchmark test split
dataset = load_dataset("ilsp/GR-ProtipaExams", split="test")

# Access multimodal content
sample = dataset[0]
print(sample["question"])
if sample["reference"] == "multimodal":
    sample["images"][0].show()
```

---

## Local Data Artifacts & Metadata

For researchers working locally or using the source repository, the data is available in several formats with additional internal metadata for traceability.

### **Internal Metadata (Excel/JSON Only)**

While the public benchmark is polished for evaluation, the internal artifacts contain pointers to facilitate local file management:

| Key | Description |
|-----|-------------|
| **image_urls** | Basemate pointers of the source image file(s) (e.g., `math_2025_Q1.png`). Used to cross-reference with the `data/` folder. |
| **q_id (Raw)** | Preserves original formatting from the source files (including float-like identifiers like `2.1`). |

### **Repository Contents**

- **JSON Files**: Individual question objects with full provenance metadata.
- **MD Files**: Markdown versions of the correct answers designed for human review.
- **Excel Master**: A consolidated file (`protipa_exams_dataset.xlsx`) containing the full dataset with AutoFilters and local path pointers.

---

## Dataset Characteristics

### **1. Subject Coverage**

| Subject | Question Types | Domain Overview |
|---------|----------------|-----------------|
| **Greek Language** | MC, T/F, Gaps, Matching | Strong emphasis on reading comprehension and syntax. |
| **Mathematics** | Open-ended, MC | Logic, geometry, and problem-solving. |
| **Physics** | Open-ended | Scientific reasoning and numeric calculation. |
| **Religious Studies**| Multiple-Choice | General knowledge and conceptual understanding. |

### **2. Known Data Gaps**

**Note on Data Gaps**: Due to the nature of public records, some limitations apply:
- **Missing Points**: Point values are only populated in ~30% of rows (primarily 2013-2019 and 2025).
- **Missing Year**: The year 2015 is currently missing as source files were publicly unavailable.
- **Missing Solutions**: Lyceum papers for 2014 (Greek) and 2018 (Greek/Math) lack official solution keys.

---

## Getting Started (Developer)

1. **Install**: `uv sync`
2. **Configure**: Set `HF_TOKEN` and `HF_REPO_ID` in your `.env`.
3. **Consolidate**: `uv run scripts/manage.py consolidate_new --extended`
4. **Push**: `uv run scripts/manage.py push --extended --with-images`


