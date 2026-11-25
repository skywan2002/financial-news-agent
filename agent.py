
import json
import requests


# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:4b"


def analyze_news(news_text):
    """
    Analyze financial news using a local LLM.
    Return the analysis as a Python dictionary.
    """

    prompt = f"""
You are a financial news analyst.

Analyze the following financial news:

{news_text}

Return ONLY a valid JSON object with these fields:

{{
    "category": "Monetary Policy, Macroeconomics, Corporate, Geopolitics, or Crypto",
    "sentiment": "Positive, Neutral, or Negative",
    "assets": ["list of affected assets"],
    "risk_level": "Low, Medium, or High",
    "summary": "A concise summary of the news",
    "impact_rationale": "Explain the potential impact on affected assets"
}}

Rules:
- Return valid JSON only.
- Do not use Markdown code fences.
- Do not invent information that is not in the article.
- If no asset is explicitly identifiable, return an empty list.
- Distinguish between general news sentiment and potential asset impact.
- Do not claim that news caused an asset price movement unless the article explicitly establishes this.
- Separate reported facts from potential implications.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        # Convert JSON string into a Python dictionary
        analysis = json.loads(result["response"])

        return analysis

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing LLM response: {e}")
        return None


if __name__ == "__main__":

    news = """
    Nvidia shares rose after the company announced stronger-than-expected
    quarterly revenue, driven by continued demand for AI chips.
    """

    analysis = analyze_news(news)

    if analysis is not None:

        print("\n=== Financial News Analysis ===\n")

        print(json.dumps(
            analysis,
            indent=4,
            ensure_ascii=False
        ))

        print("\n=== Individual Fields ===")

        print("Category:", analysis["category"])
        print("Sentiment:", analysis["sentiment"])
        print("Assets:", analysis["assets"])
        print("Risk Level:", analysis["risk_level"])
