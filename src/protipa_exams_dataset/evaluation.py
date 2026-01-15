import logging
import time
import traceback
import lm_eval
from lm_eval.models.openai_completions import OpenAIChatCompletion
from lm_eval.tasks import ConfigurableTask

logger = logging.getLogger(__name__)

def run_evaluation(model_name, api_base, task_dict, eval_limit=None):
    """
    Runs evaluation for a specific model using lm_eval.
    """
    logger.info(f"Starting evaluation for model: {model_name}")
    try:
        # Construct endpoint URL
        chat_api_url = api_base
        if not chat_api_url.endswith("/chat/completions"):
            chat_api_url = chat_api_url.rstrip("/") + "/chat/completions"

        model = OpenAIChatCompletion(
            model=model_name,
            base_url=chat_api_url,
            num_fewshot=0,
            eos_string="<|end_of_text|>",
            max_retries=10,
            num_concurrent=1
        )

        results = lm_eval.evaluate(
            lm=model,
            task_dict=task_dict,
            limit=eval_limit,
            apply_chat_template=True
        )
        return results

    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        logger.error(traceback.format_exc())
        return None
