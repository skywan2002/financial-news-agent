
import os
import json

from dotenv import load_dotenv
from google import genai

from tools import normalize_assets, validate_news


# Load environment variables from .env
load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"


# JSON Schema for structured output
NEWS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "category": {
            "type": "STRING",
            "enum": [
                "Monetary Policy",
                "Macroeconomics",
                "Corporate",
                "Geopolitics",
                "Crypto"
            ]
        },
        "sentiment": {
            "type": "STRING",
            "enum": ["Positive", "Neutral", "Negative"]
        },
        "assets": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": [
                    "BTC",
                    "ETH",
                    "Gold",
                    "S&P 500",
                    "NASDAQ",
                    "USD",
                    "US Treasury",
                    "Nvidia"
                ]
            }
        },
        "risk_level": {
            "type": "STRING",
            "enum": ["Low", "Medium", "High"]
        },
        "summary": {
            "type": "STRING"
        },
        "impact_rationale": {
            "type": "STRING"
        }
    },
    "required": [
        "category",
        "sentiment",
        "assets",
        "risk_level",
        "summary",
        "impact_rationale"
    ],
    "additionalProperties": False
}


def analyze_news(news_text):
    """
    Analyze financial news using the Gemini API.
    Return the analysis as a Python dictionary.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return None

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are a professional financial news analyst.

Analyze the following financial news:

{news_text}

Category Classification Rules:
- Monetary Policy: News about central bank policies, interest rate decisions, monetary policy changes, and central bank actions.
- Macroeconomics: News about inflation, GDP, employment, economic growth, currencies, government bonds, and broader economic conditions.
- Corporate: News about individual companies, corporate earnings, business operations, mergers and acquisitions, and company-specific developments.
- Geopolitics: News about international conflicts, political tensions, trade wars, tariffs, sanctions, and geopolitical events.
- Crypto: News about cryptocurrencies, blockchain, digital assets, and cryptocurrency markets.
- Select the single most relevant category based on the main focus of the article.
- If an article covers multiple topics, classify it according to its primary focus.

Sentiment Classification Rules:
- Positive: The news indicates favorable developments or potentially beneficial implications for the affected assets or financial markets.
- Negative: The news indicates unfavorable developments or potentially adverse implications for the affected assets or financial markets.
- Neutral: The news is primarily factual, contains balanced or mixed implications, or has no clearly identifiable positive or negative financial impact.
- Determine sentiment based on the overall financial implications of the news, not isolated positive or negative words.
- Evaluate sentiment from the perspective of the affected assets or financial markets.

Risk Level Classification Rules:
- Low: The news has limited potential financial impact, involves routine developments, or is unlikely to create significant market disruption.
- Medium: The news may significantly affect a specific company, industry, asset, or market segment, but its broader market impact is likely to be limited.
- High: The news may cause substantial market volatility, systemic financial risks, or widespread disruption across financial markets.
- Assess risk based on the potential severity and scope of financial consequences, not simply whether the news is positive or negative.
- Do not assign High risk unless there is a reasonable basis for expecting significant or widespread financial consequences.

Asset Classification Rules:
- Use only the supported standardized asset names.
- Do not invent alternative asset names.
- Distinguish between stock exchanges and market indices.
- If an asset cannot be confidently mapped to a supported asset, omit it.
- Return an empty list if no supported assets are affected.

General Rules:
- Do not invent information that is not in the article.
- Distinguish between general news sentiment and potential asset impact.
- Do not claim that news caused an asset price movement unless the article explicitly establishes this.
- Separate reported facts from potential implications.
- Apply sentiment and risk classification criteria consistently.
- Sentiment and risk level must be evaluated independently.
- Positive news can have High risk, and Negative news can have Low risk.
- Keep the summary concise and factual.
- Explain the potential financial implications in impact_rationale.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_json_schema": NEWS_SCHEMA
            }
        )

        if not response.text:
            print("Error: Gemini returned an empty response.")
            return None

        # Convert JSON string into a Python dictionary
        analysis = json.loads(response.text)

        # Normalize asset names
        analysis["assets"] = normalize_assets(analysis["assets"])

        # Validate the analysis
        is_valid, error = validate_news(analysis)

        if not is_valid:
            print(f"Invalid news analysis: {error}")
            return None

        return analysis

    except json.JSONDecodeError as e:
        print(f"Error processing Gemini JSON response: {e}")
        return None

    except Exception as e:
        print(f"Error connecting to Gemini API: {e}")
        return None


if __name__ == "__main__":

    news = """
Dollar steady as sliding yields temper safe-haven bid

Thu 26/11/2026 - 02:43
SHANGHAI, Nov 26 (Reuters) - The dollar was steady in Asian trading on Thursday after dropping to a one-week low overnight, as falling Treasury yields and fragile risk sentiment offset the greenback's usual safe-haven appeal ahead of the Thanksgiving holiday.

The US dollar index slipped overnight as US equities pared losses and showed little reaction to strong economic data, leaving it well off a 19-month high struck on Monday when Jensen Huang's return as Nvidia chief executive reignited the artificial intelligence trade, while also bolstering confidence that robust US economic growth could warrant further rate hikes.

"The risk-reward of chasing further USD gains looks less attractive, but hawkish Fed risks should keep the dollar supported," said Sim Moh Siong, an FX strategist at OCBC.

He said that the dollar's relationship with equities could remain choppy, caught between safe-haven demand and concerns that a deeper AI slowdown could weigh on US growth.

The yen edged 0.1% higher against the dollar, trading around 159.06 in early Asian trade. The euro was little changed at $1.097 as the market's focus turned to the accounts of the October European Central Bank meeting, scrutinised for hints on the rate outlook.

The US dollar index, which measures the greenback's strength against a basket of six currencies, was flat at 103.7.

Hotter-than-expected US core personal consumption expenditures inflation in October, a sharp upward revision to third-quarter gross domestic product and a Federal Reserve survey showing continued economic growth and sticky inflation cemented expectations the Fed will raise rates by 25 basis points (bps) in December and again in early 2027.

"There may also be a long weekend effect where being long bonds might not be a bad option," ING analysts said in a note. "But still, it's been a week so far in which Treasuries have ignored valid reasons for yields to rise."

Treasury yields fell across the curve, led by the long end. The 10-year yield dropped as much as 10 bps to a 2-1/2-month low of 4.8%.

Falling yields handed some relief to beaten-down currencies. The Aussie dollar recovered above 0.66 and was last at 0.6635, bouncing off an earlier low of 0.6557 that marked its weakest level in almost a year.

Adding to the sour tone, the Trump administration proposed new tariffs on goods from 14 major trading partners on Wednesday. The trade levies are meant to combat what Washington called structural excess industrial capacity, with the highest rates to hit China, which the US blames most for flooding world markets with cheap exports.

With most US markets shut on Thursday for Thanksgiving, Asia trades without fresh US price action, leaving thin liquidity for the session, market participants said.

The Swiss franc edged lower on Thursday, having firmed 0.5% overnight on safe-haven demand, while the Canadian dollar held on to a two-week high as oil prices rose.

Brent crude futures settled 3.5% higher at $94.61 a barrel, after Yemen's Houthis said they had attacked ​Saudi Arabia's largest oil refinery.


(Reporting by Shanghai Newsroom; Editing by Thomas Derpinghaus)
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
