import json
import re
from inspect_ai.scorer import scorer, Score
from inspect_ai.model import get_model

@scorer(metrics=[])
def generic_judge_scorer(instructions: str, model: str | None = None):
    """
    A scorer that extracts subject-specific rubrics 
    from sample metadata, prompts the judge model, and parses the JSON response.
    """
    async def score(state, target):
        # 1. Get the grader model instance
        import os
        grader_model_id = os.environ.get("GRADER_MODEL_ID")
        grader_base_url = os.environ.get("GRADER_BASE_URL")
        grader_api_key = os.environ.get("GRADER_API_KEY")

        if grader_model_id and grader_base_url:
            # Explicitly route to the grader backend bypassing global OPENAI_BASE_URL
            grader = get_model(
                f"openai/{grader_model_id}",
                base_url=grader_base_url,
                api_key=grader_api_key
            )
        else:
            grader = get_model(model) if model else get_model(role="grader")
        
        # 2. Extract rubric from sample metadata (fallback to default instructions)
        rubric = state.metadata.get("grading_instructions") or instructions
        
        # 3. Build the grading prompt
        prompt = (
            "You are assessing a submitted answer on a given task based on a criterion.\n\n"
            "[BEGIN DATA]\n"
            f"[Task]: {state.input}\n"
            f"[Submission]: {state.output.completion}\n"
            f"[Criterion]: {target.text}\n"
            "[END DATA]\n\n"
            f"{rubric}\n\n"
            "Format your response as a JSON object with 'grade' and 'explanation' fields:\n"
            '{\n  "grade": 1.0 or 0.0,\n  "explanation": "Brief explanation of the grade"\n}'
        )
        
        # 4. Call the model (normal text generation)
        result = await grader.generate(
            input=prompt
        )
        
        completion = result.completion.strip()
        
        # 5. Extract JSON block programmatically
        grade = 0.0
        explanation = completion
        
        # Attempt to find JSON object in the completion
        json_match = re.search(r"\{.*\}", completion, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                grade = float(data.get("grade", 0.0))
                explanation = data.get("explanation", completion)
                return Score(
                    value=grade,
                    explanation=explanation,
                    metadata={"raw_completion": completion}
                )
            except Exception:
                pass
                
        # 6. Fallback parser if JSON parsing fails
        # Look for numeric grade (1.0 or 0.0 or 1 or 0)
        grade_match = re.search(r"\b(1\.0|0\.0|1|0)\b", completion)
        if grade_match:
            val = float(grade_match.group(1))
            grade = 1.0 if val in [1.0, 1] else 0.0
            
        return Score(
            value=grade,
            explanation=explanation,
            metadata={"raw_completion": completion}
        )
            
    return score
