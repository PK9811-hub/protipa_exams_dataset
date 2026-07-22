import os
from openai import OpenAI
from dotenv import load_dotenv

# Φορτώνει αυτόματα τα κλειδιά από το .env αρχείο σου
load_dotenv()

# Παίρνουμε το κλειδί του OpenRouter
or_api_key = os.getenv("OPENROUTER_API_KEY")

# Αρχικοποίηση του client 
client = OpenAI(
    api_key=or_api_key,
    base_url=os.getenv("OPENROUTER_BASE_URL")
)

def test_model(model_name, prompt="Γεια σου! Γράψε μου μόνο τη λέξη 'Λειτουργώ' στα Ελληνικά."):
    print(f"\n--- Δοκιμή μοντέλου: {model_name} ---")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.1
        )
        print("✅ ΕΠΙΤΥΧΙΑ! Η απάντηση του μοντέλου:")
        print(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"❌ ΣΦΑΛΜΑ κατά την κλήση του {model_name}:")
        print(e)

if __name__ == "__main__":
    qwen_model = "qwen/qwen-2.5-coder-32b-instruct"
    print("Ξεκινάει ο έλεγχος σύνδεσης...")
    test_model(qwen_model)