# Prot-Ex Dataset

Prot-Ex ([ilsp/greek-protipa-exams](https://huggingface.co/datasets/ilsp/greek-protipa-exams)) is a dataset derived from publicly available exam questions and solutions used for student admission to Model and Experimental Schools (Πρότυπα και Πειραματικά Σχολεία) in Greece. 

Spanning from 2013 to 2026, the dataset includes questions with the following features:

*   **Subjects**: Greek Language, Mathematics, Physics, and Religious Studies
*   **Educational Levels**: Questions targeted at both Gymnasium (Γυμνάσιο) and Lyceum (Λύκειο) admission exams.
*   **Formats**: Multiple Choice, True/False, Matching, Fill-in-the-Gaps, and Open-Ended questions.
*   **Modalities**: Questions suitable for multimodal evaluation, featuring high-fidelity images/diagrams, LLM-genarated image descriptions, and OCR transcriptions.

The benchmark can be used for the evaluation of LLMs on complex, multi-subject, multi-format questions in the Greek language. Additionally, it may be useful as a high-quality resource for quantitative educational research.

## 🔍 Note 

The complete dataset is officially hosted and maintained on the **Hugging Face Hub**.

To facilitate quick inspection of the dataset's structure, formatting, and metadata directly from this repository, we have also included a consolidated export: **[protipa_exams_public.xlsx](./protipa_exams_public.xlsx)**.

Please note: Both the public Hugging Face repository and this Excel file purposefully exclude the **2019** exam data. This specific year is used as a private holdout test set to ensure rigorous, contamination-free model evaluation, as detailed in our preprint.

## Dataset Creation

The source material was extracted from the official portal of the **Governing Body of Model and Experimental Schools** ([https://depps.minedu.gov.gr/](https://depps.minedu.gov.gr/)). It was then converted into the current format via specialized processing pipelines by the authors.

**Disclaimer**: While every effort has been made to ensure the accuracy and completeness of this structured dataset, any errors, omissions, or formatting issues are the result of the processing and transformation pipeline and are **not related** to the original source or the Ministry of Education.

## Dataset Structure


| Column | Description |
|--------|-------------|
| **id** | Unique identifier. |
| **subject** | Academic subject (including `greek_language`, `mathematics`, `physics`, `religious studies`). |
| **format** | `multiple_choice`, `true_false`, `matching`, `fill_in_the_gaps`, `open_ended`. |
| **reference** | Additional reference inputs: `none`, (text) `passage`, `multimodal`, `table`. |
| **question** | The core question. |
| **input** | Passage. |
| **images** | Visual asset(s) (diagrams, photos, geometrical figures). |
| **choices** | Candidate answers for closed-ended questions. |
| **answer_text** | The answer/solution (string). |
| **answer_index** | The index of the correct answer for multiple choice questions. |
| **image_description** | LLM-generated textual descriptions of visual assets. |
| **image_transcription** | OCR/Text extraction from within the visual assets. |
| **points** | Assigned point value for the question (numeric). May be null if missing in the source. |
| **year** | The year of the exam. |
| **admission_level** | The target education level: `gymnasium` or `lyceum`. |
| **exam_set** | The ID of the exam batch/paper (e.g., 1, 2) |
| **q_id** | The specific ID/number of the question within its set. |

## Usage 

```python
import random

# Load a random sample
random_idx = random.randint(0, len(dataset) - 1)
sample = dataset[random_idx]

print(f"Sample Index: {random_idx} | ID: {sample['id']}")
print(f"Question: {sample['question']}\n")

# 1. Handle Multimodal Metadata (Descriptions & Transcriptions)
if sample.get('image_description'):
    print(f"🖼️  Image Description: {sample['image_description']}")

if sample.get('image_transcription'):
    print(f"📝 Image Transcription: {sample['image_transcription']}")

# 2. Handle Choices
if sample.get('choices'):
    print("\nChoices:")
    correct_idx = sample.get('answer_index')
    for i, choice in enumerate(sample['choices']):
        marker = "[✅]" if i == correct_idx else "[  ]"
        print(f"  {marker} {i}: {choice}")
    print(f"\nCorrect Answer Index: {correct_idx}")
else:
    print(f"\nAnswer: {sample['answer_text']}")

# ---------------------------------------------------------
# Output Example:
# Sample Index: 123 | ID: math_lyc_2022_1_37
# Question: Στο σχήμα τα τετράγωνα του πλέγματος έχουν πλευρά μήκους 2 cm. Η περίμετρος του τριγώνου ΑΒΓ είναι ίση με:
# 
# 🖼️  Image Description: The image displays an isosceles triangle labeled $AB\Gamma$ drawn on a square grid.\nGrid: The background consists of a regular grid of squares.\nTriangle Vertices:\nVertex $A$ is at the top center.\nVertex $B$ is at the bottom left.\nVertex $\Gamma$ is at the bottom right.\nDimensions based on Grid Units:\nThe base $B\Gamma$ spans 4 horizontal grid units.\nThe height of the triangle (vertical distance from base $B\Gamma$ to vertex $A$) spans 4 vertical grid units.\nThe vertex $A$ is horizontally centered between $B$ and $\Gamma$ (2 units from $B$, 2 units from $\Gamma$).
# 
# Choices:
#   [✅] 0: A. $4(\sqrt{5}+1)$ cm
#   [  ] 1: B. 10 cm
#   [  ] 2: Γ. 8 cm
#   [  ] 3: Δ. $4\sqrt{3}$ cm
# 
# Correct Answer Index: 0
# 
```

## Known Data Gaps

- **Missing Year**: The year 2015 is currently missing as source files were unavailable.
- **Missing Points**: Point values are only available for ~30% of rows.

---

## Local Data

For researchers working locally or using the source repository, the data is available in several formats with additional internal metadata for traceability.

### Repository Contents

- **Excel Export**: A consolidated file (`protipa_exams_public.xlsx`) is provided in this repository, containing the public split of the dataset.
- **notebooks/**: Jupyter notebooks used for exploratory data analysis, pipeline testing, and evaluation setup.
- **scripts/**: Utility scripts for data management, validation, and automated publishing to the Hugging Face Hub.
- **src/**: The core source code module, containing custom data loaders and the evaluation logic (e.g., Inspect AI configurations).

---

## Dataset Characteristics

### 1. Subject Coverage

| Subject | Question Types | Domain Overview |
|---------|----------------|-----------------|
| **Greek Language** | MC, T/F, Gaps, Matching | Strong emphasis on reading comprehension and syntax. |
| **Mathematics** | Open-ended, MC | Logic, geometry, and problem-solving. |
| **Physics** | Open-ended | Scientific reasoning and numeric calculation. |
| **Religious Studies**| Multiple-Choice | General knowledge and conceptual understanding. |

### 2. Known Data Gaps

**Note on Data Gaps**: Due to the nature of public records, some limitations apply:
- **Missing Points**: Point values are only populated in ~30% of rows.
- **Missing Year**: The year 2015 is currently missing as source files were publicly unavailable.
- **Missing Solutions**: Lyceum papers for 2014 (Greek) and 2018 (Greek/Math) lack official solution keys.

---


## Benchmarking & Evaluation

This repository provides ready-to-use configurations for evaluating Large Language Models on the dataset, supporting different task modalities.

### 1. LM-Eval Harness 
All tasks callable via the Language Model Evaluation Harness (`lm-eval`) can be found in the [`tasks/greek_protipa_exams`](https://github.com/PK9811-hub/protipa_exams_dataset/tree/dev/tasks/greek_protipa_exams) directory. 
- Includes configurations to run experiments on **closed-ended**, **open-ended**, and **structured** (single-word/short phrase) questions.
- Tasks are organized both as broad aggregates (e.g., `closed_aggregate`, `open_aggregate`, `structured_aggregate`) and as fine-grained, subject-specific tasks (e.g., Mathematics, Greek Language).

### 2. Inspect AI (LLM-as-a-Judge)
For open-ended questions where exact text matching is insufficient, we provide configurations for the Inspect AI framework.
- Located in the [`protipa_exams_dataset/evals`](https://github.com/PK9811-hub/protipa_exams_dataset/tree/dev/src/protipa_exams_dataset/evals) directory.
- Contains the evaluation configurations (folders), custom evaluation prompts, and `.py` scoring scripts.
- Implements an **LLM-as-a-judge** paradigm, utilizing a secondary model to assess the logical flow, correctness, and reasoning capabilities of the generated answers rather than superficial lexical overlap.

#### Running Evaluations

<details>
<summary>Running Inspect AI Evaluations</summary>

Make sure your `.env` file is set up with the correct variables (e.g., API keys and model paths). First, load the environment variables:

```bash
export $(grep -v '^#' .env | xargs)
```

The evaluation script is highly flexible. You can evaluate the entire dataset, or filter it down using standard Inspect AI flags (like --limit for quick testing) and custom task parameters (-T).

Filtering Rules & Task Parameters:

* Metadata Filtering: Use -T filter_field and -T filter_value to specify metadata columns and their allowed values.

* AND Logic: Use ; to apply multiple filters simultaneously (e.g., -T filter_field="subject;format").

* OR Logic: Use , to allow multiple values for a single field (e.g., -T filter_value="physics;open_ended,fill_in_the_gaps").

* Strict Open-Ended: Use -T filter_field="choices" and -T filter_value="empty" to exclusively evaluate questions that do not have multiple-choice options. To evaluate only questions WITH options, use a value like "not_empty".

* Image Descriptions: Use -T filter_field="has_image_description" and -T filter_value="true" to only evaluate questions that contain image descriptions.

* Few-Shot Prompting: Use -T num_fewshot=N (where N is the number of examples) to add few-shot examples from the dev split. Set to 0 for Zero-Shot.

Here are some examples of how to run the evaluations:

**Example 1: Strict Open-Ended Evaluation (Zero-Shot)**

```bash
uv run inspect eval src/protipa_exams_dataset/evals/tasks.py \
  --model "openai/$MODEL_ID" \
  --max-connections 10 \
  -T num_fewshot=0 \
  -T dataset_path="$HF_REPO_ID" \
  -T split=test \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field="format;subject;choices" \
  -T filter_value="open_ended,matching,fill_in_the_gaps;physics;empty" \
  -T grader_model="openai/$GRADER_MODEL_ID" \
  --batch false
```

**Example 2: Quick Test with Few-Shot Prompting & Limit**

```bash
uv run inspect eval src/protipa_exams_dataset/evals/tasks.py \
  --model "openai/$MODEL_ID" \
  --limit 5 \
  --max-connections 10 \
  -T num_fewshot=3 \
  -T dataset_path="$HF_REPO_ID" \
  -T split=test \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field="subject" \
  -T filter_value="greek_language" \
  -T grader_model="openai/$GRADER_MODEL_ID" \
  --batch false
```

**Example 3: Evaluating Questions with Image Descriptions**

```bash
uv run inspect eval src/protipa_exams_dataset/evals/tasks.py \
  --model "openai/$MODEL_ID" \
  --max-connections 10 \
  -T num_fewshot=0 \
  -T dataset_path="$HF_REPO_ID" \
  -T split=test \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field="subject;has_image_description" \
  -T filter_value="mathematics;true" \
  -T grader_model="openai/$GRADER_MODEL_ID" \
  --batch false
```
</details>

## Getting Started (Developer)

### Prerequisites

*   **Environment**: Python 3.10+ (Recommended: use `uv` for dependency management).
*   **API Tokens**: Create a `.env` file in the root directory with your Hugging Face credentials:
    ```bash
    HF_TOKEN=your_huggingface_write_token
    HF_REPO_ID=ilsp/greek-protipa-exams 
    ```

### Management Commands

The repository includes a comprehensive management script `scripts/manage.py` to handle the data lifecycle.

#### 1. Data Consolidation (Local)
Transforms raw JSON/Markdown files into a structured Excel master file.
*   **Standard**: `uv run scripts/manage.py consolidate`
*   **Extended (with Schema Tags)**: Adds `format` and `reference` columns based on linguistic markers.
    ```bash
    uv run scripts/manage.py consolidate --extended
    ```

#### 2. Data Validation
Before pushing to the Hub, verify that all multimodal assets (images) referenced in the JSON files exist on disk:
```bash
python scripts/check_images.py
```

#### 3. Pushing to Hugging Face Hub
Synchronizes the local dataset with the Hugging Face repository. 

*   **Text only**: `uv run scripts/manage.py push --extended`
*   **Multimodal (Includes Images)**: Embeds the actual pixel data into the Parquet files.
    ```bash
    uv run scripts/manage.py push --extended --with-images
    ```

---
