import os
import traceback
import lm_eval
from lm_eval.models.openai_completions import OpenAIChatCompletion
from lm_eval.tasks import ConfigurableTask
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

def run_evaluation(model_name, api_base=None, task_dict=None, eval_limit=None):
    """
    Runs evaluation for a specific model using lm_eval.
    """
    logger.info(f"Starting evaluation for model: {model_name}")
    try:
        # Use provided api_base or fall back to environment variable
        chat_api_url = api_base or os.getenv("OPENAI_BASE_URL")
        
        if not chat_api_url:
            raise ValueError("No API base URL provided. Set OPENAI_BASE_URL in .env or pass it as an argument.")

        if not chat_api_url.endswith("/chat/completions"):
            chat_api_url = chat_api_url.rstrip("/") + "/chat/completions"

        # Note: OpenAIChatCompletion automatically looks for OPENAI_API_KEY env var
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
