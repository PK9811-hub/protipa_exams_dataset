📚 **GR-ProtipaExams Dataset: Structured Exam Questions (2013-2025)**

**Overview**

The GR-ProtipaExams project introduces a comprehensive, structured dataset, called “GR-ProtipaExams”, derived from publicly available exam questions and solutions used for student admission to Model and Experimental Schools (Protipa and Peiramatika Schools) in Greece. This dataset spans the years 2013 to 2025 and covers core secondary education subjects.

The primary goal is to provide a standardized resource for educational research, quantitative statistical analysis, and, particularly, for training, finetuning, and evaluating Large Language Models (LLMs) on complex, multi-subject assessment tasks in the Greek language.

---

## 🚀 Hugging Face Benchmark Schema

This is the primary public version of the dataset, optimized for machine learning evaluation and high-speed browsing. It follows a **Narrative Flow** to provide immediate context for researchers.

### **Dataset Columns**

| Column | Description |
|--------|-------------|
| **id** | Standardized unique identifier for the row entry (breadcrumb format). |
| **subject** | The academic subject (e.g., `greek_language`, `mathematics`, `physics`, `religious studies`). |
| **format** | **(Structural)** Interaction format: `multiple_choice`, `true_false`, `matching`, `fill_in_the_gaps`, `open_ended`. |
| **reference** | **(Structural)** Contextual requirement/addenda: `passage`, `multimodal`, `table`, `none`. |
| **question** | The introductory text/core task. |
| **input** | Supplementary text (passages, diagram descriptions). |
| **images** | **(Visual)** The actual image asset(s) rendered as pixel data. |
| **choices** | Candidate answers for closed-ended questions. |
| **answer_text** | The processed, validated answer/solution (string). |
| **answer_index** | The numeric mapping of the answer (for programmatic evaluation). |
| **image_description** | Detailed textual proxy for visual content. |
| **image_transcription** | OCR/Text extraction from within the visual assets. |
| **points** | Assigned point value for the question (numeric). |
| **year** | The year of the exam. |
| **admission_level** | The target education level: `gymnasium` or `lyceum`. |
| **exam_set** | The ID of the exam batch/paper (e.g., 1, 2). |
| **q_id** | The specific ID/number of the question within its set. |

### **Usage (Hugging Face)**

```python
from datasets import load_dataset

# Load the benchmark test split
dataset = load_dataset("PK9811/gr-protipa-exams", split="test")

# Access multimodal content
sample = dataset[0]
print(sample["question"])
if sample["reference"] == "multimodal":
    sample["images"][0].show()
```

---

## 📁 Local Data Artifacts & Metadata

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

## 📊 Dataset Characteristics

### **1. Subject Coverage**

| Subject | Question Types | Domain Overview |
|---------|----------------|-----------------|
| **Greek Language** | MC, T/F, Gaps, Matching | Strong emphasis on reading comprehension and syntax. |
| **Mathematics** | Open-ended, MC | Logic, geometry, and problem-solving. |
| **Physics** | Open-ended | Scientific reasoning and numeric calculation. |
| **Religious Studies**| Multiple-Choice | General knowledge and conceptual understanding. |

### **2. Known Data Gaps**

⚠️ **Note on Data Gaps**: Due to the nature of public records, some limitations apply:
- **Missing Points**: Point values are only populated in ~30% of rows (primarily 2013-2019 and 2025).
- **Missing Year**: The year 2015 is currently missing as source files were publicly unavailable.
- **Missing Solutions**: Lyceum papers for 2014 (Greek) and 2018 (Greek/Math) lack official solution keys.

---

## 🏗️ Schema Evolution & Rationale

We introduced the **Structural Duo** to shift the dataset from a simple "data dump" to a **diagnostic benchmark**.

### **The Structural Duo**

1. **`format`** (The Interaction Model): Defines *how* the model must interact with the answer space (e.g., `matching` vs `multiple_choice`).
2. **`reference`** (The Data Requirement): Identifies *which* additional asset type (addenda) is required. This allows for pinpointing failure points: Is it a reading error (`passage`) or a vision error (`multimodal`)?

### **Deterministic Heuristics**
Instead of black-box AI tagging, we use consistent, rule-based heuristics:
- `len(input) > 200` $\rightarrow$ triggers **`passage`**.
- Subject mapping + Regex $\rightarrow$ triggers **`format`**.

---

## 🛠️ Getting Started (Developer)

1. **Install**: `uv sync`
2. **Configure**: Set `HF_TOKEN` and `HF_REPO_ID` in your `.env`.
3. **Consolidate**: `uv run scripts/manage.py consolidate_new --extended`
4. **Push**: `uv run scripts/manage.py push --extended --with-images`
