# Dev Split Coverage per Task

Here is the breakdown of the **20 dev questions** from the `few_shot_dev` dataset across the tasks defined by the task YAML configurations:

| Task / YAML File | Task Name in YAML | Mode | Subject | Dev Questions Count |
| :--- | :--- | :--- | :--- | :---: |
| **`language_closed.yaml`** | `greek_protipa_exams_language_closed` | Closed | Greek Language | **0** |
| **`maths_closed.yaml`** | `greek_protipa_exams_maths_closed` | Closed | Mathematics | **0** |
| **`religious_studies_closed.yaml`** | `greek_protipa_exams_religious_studies_closed` | Closed | Religious Studies | **0** |
| **`language_open.yaml`** | `greek_protipa_exams_language_open` | Open | Greek Language | **5** |
| **`maths_open.yaml`** | `greek_protipa_exams_maths_open` | Open | Mathematics | **5** |
| **`physics_open.yaml`** | `greek_protipa_exams_physics_open` | Open | Physics | **0** |
| **`language_structured.yaml`** | `greek_protipa_exams_language_structured` | Structured | Greek Language | **10** |

## Summary
* **No dev questions exist** for any of the **Closed** tasks (`language_closed`, `maths_closed`, `religious_studies_closed`) or the **Physics Open** task (`physics_open`). Done [x]
* The **20 dev questions** are fully distributed among three tasks:
  * **`greek_protipa_exams_language_open`** (5 items)
  * **`greek_protipa_exams_maths_open`** (5 items)
  * **`greek_protipa_exams_language_structured`** (10 items: 5 fill-in-the-gaps and 5 matching)
