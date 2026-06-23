# Inspect AI Evaluation Framework & Protipa Dataset Integration

This is an outline of the Inspect AI evaluation framework, the integration for the Protipa exams dataset, execution instructions, result inspection, and a roadmap of tasks.

---

## 1. How Inspect AI Works

An Inspect evaluation is a `Task` that integrates three components:

1. **Dataset**: A collection of samples. Each `Sample` contains:
   - `input`: The prompt or list of messages for the model.
   - `target`: The reference answer.
   - `metadata`: Key-value pairs for routing or scoring.
2. **Solver (Plan)**: A pipeline of steps applied to samples to produce answers.
   - **Structure**: A plan is a Python list of solver functions.
   - **Execution**: Solvers run sequentially to modify the sample state. The `generate()` solver calls the model API.
3. **Scorer**: The metric that grades model answers against targets.
   - **Built-in**: Includes exact match, inclusion, multiple choice, and model grading.
   - **Custom**: A function registered with the `@scorer` decorator that returns a `Score` containing a value, answer, and explanation.

---

## 2. Protipa Dataset Integration

The evaluation structure is implemented under `src/protipa_exams_dataset/evals/`:

* **Task (`tasks.py`)**: A parameterized function `generic_evaluation` that loads Hugging Face splits, processes input/context columns, and filters samples by fields (such as `subject` or `format`).
* **Subject Rubrics**: Grading prompts (for Physics, Mathematics, and Greek Language) are resolved using the sample's `subject` field and passed to the scorer in the sample's `metadata`.
* **Curriculum Constraints**:
  - **Persona**: The model under evaluation receives system instructions to write as a Greek student (enforcing brevity and Greek language output).
  - **Banning English**: The judge model is instructed to award a `0.0` score if the student model includes English terminology.
  - **Vocabulary Constraints**: Enforces Greek school terms (`αγωγή` for conduction, `μεταφορά` for convection) and penalizes alternative translations (`σύμπαση` or `μόλυνση`).
* **JSON Scorer (`scorers.py`)**: An async scorer that prompts the judge model to return a JSON object (`{"grade": 1.0/0.0, "explanation": "..."}`). A Python parser extracts these fields and includes a fallback parser.

---

## 3. How to Run Evaluations (tested on linux)

Export environment variables from `.env` before running:

```bash
# 1. Export env variables
export $(grep -v '^#' .env | xargs)

# 2. Run open-ended questions (limited to 100 samples)
PYTHONPATH=src uv run inspect eval src/protipa_exams_dataset/evals/tasks.py \
  --model $MODEL_ID \
  --limit 100 \
  -T dataset_path=$HF_REPO_ID \
  -T split=test \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field=format \
  -T filter_value=open_ended \
  -T grader_model=openai/$MODEL_ID

# 3. Run a single sample (e.g., Physics)
PYTHONPATH=src uv run inspect eval src/protipa_exams_dataset/evals/tasks.py \
  --model $MODEL_ID \
  -T dataset_path=$HF_REPO_ID \
  -T split=test \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field=id \
  -T filter_value=phys_gym_2014_1_1 \
  -T grader_model=openai/$MODEL_ID

# 4. Run evaluation (grader model is dynamically routed via environment variables)
PYTHONPATH=src uv run inspect eval src/protipa_exams_dataset/evals/tasks.py \
  --model $MODEL_ID \
  --limit 5 \
  -T dataset_path=$HF_REPO_ID \
  -T split=test \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field=format \
  -T filter_value=open_ended \
  --temperature ${MODEL_TEMPERATURE:-0.3} \
  --top-p ${MODEL_TOP_P:-0.95} \
  --top-k ${MODEL_TOP_K:-64}
```

---

## 4. How to Inspect Results

### Option A: VS Code Extension (Recommended)
Use the **Inspect AI** VS Code extension to view results inside the editor:
1. Install the "Inspect AI" extension in VS Code.
2. The extension displays evaluation logs and provides:
   - A results panel alongside the source code.
   - A Tasks panel to browse, run, and debug tasks.
   - An environment configuration panel to edit `.env` settings.
3. Keyboard shortcuts:
   - Run task: `Ctrl+Shift+U` (Windows/Linux) or `Cmd+Shift+U` (Mac).
   - Debug task: `Ctrl+Shift+T` (Windows/Linux) or `Cmd+Shift+T` (Mac).

### Option B: Browser Log Viewer
Launch the log viewer from the terminal:
```bash
uv run inspect view
```
Open **`http://localhost:7575`** in the browser to:
* Inspect prompt history (system and user messages) for each sample.
* Read the judge's JSON grading explanations.
* Sort and filter samples by score to identify model failure modes.

---

## 5. TODO

### Documentation:
* [Inspect AI Tutorial](https://inspect.aisi.org.uk/tutorial.html), this document

### Key Tasks:
- [ ] **Framework Evaluation (ALL)**: Familiarize with the framework and assess suitability for open-ended questions.
- [ ] **Exam benchmarks (PK, EK)**: Execute the code, inspect results on protipa, and expand to panhellenic exams. 
- [ ] **Culture Benchmark Expansion (SM, PP)**: Same tasks expanded to the culture benchmark.
- [ ] **Prompt and Rubric Tuning (ALL)**: Fix hardcoded prompts, refine subject rubrics for more granular scoring.
- [ ] **Concurrency Tuning**: Optimize `--max-connections` and `--max-tasks` in the CLI to maximize throughput on the local vLLM server.


