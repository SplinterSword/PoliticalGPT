from dotenv import load_dotenv
from google import genai
from groq import Groq
import json
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_contradiction_news(politician_name: str):
    
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
    Find all contradictions made by {politician_name}. The response must be a JSON object formatted as follows:

    {{
      "contradictions": [
        {{
          "contradiction_id": 1,
          "topic": "Example Topic",
          "statement_1": "First statement made by the politician.",
          "statement_2": "Second, contradictory statement made by the politician.",
          "summary": "Summary of how the statements contradict each other.",
          "articles": [
            "https://example.com/article1",
            "https://example.com/article2"
          ]
        }},
        ...
      ]
    }}

    Ensure that:
    - There are at least 5 contradictions.
    - The JSON output strictly follows the given structure.
    - Each contradiction has a valid topic, statements, and a summary.
    - Each contradiction includes at least one credible article as a source.
    - Return the json data only.
    """

    try:
        chat_completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        response_text = chat_completion.choices[0].message.content
        only_json = response_text.split("```")[1]
        cleaned_json = only_json.replace("json", "").replace("```", "").strip()

        try:
            json_data = json.loads(cleaned_json)
            return json_data
        except json.JSONDecodeError:
            return f"Error: Could not parse response as JSON.\nRaw response:\n{response_text}"
    except Exception as e:
        return f"An error occurred: {e}"
