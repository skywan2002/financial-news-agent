from tools import normalize_assets, validate_news
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
    "category": ["Monetary Policy", "Macroeconomics", "Corporate", "Geopolitics", "Crypto"],
    "sentiment": ["Positive", "Neutral", "Negative"],
    "assets": ["BTC", "ETH", "Gold", "S&P 500", "NASDAQ", "USD", "US Treasury", "Nvidia"],
    "risk_level": ["Low", "Medium", "High"],
    "summary": "A concise summary of the news",
    "impact_rationale": "Explain the potential impact on affected assets"
}}

Category Classification Rules:
- Monetary Policy: News about central bank policies, interest rate decisions, monetary policy changes, and central bank actions.
- Macroeconomics: News about inflation, GDP, employment, economic growth, currencies, government bonds, and broader economic conditions.
- Corporate: News about individual companies, corporate earnings, business operations, mergers and acquisitions, and company-specific developments.
- Geopolitics: News about international conflicts, political tensions, trade wars, tariffs, sanctions, and geopolitical events.
- Crypto: News about cryptocurrencies, blockchain, digital assets, and cryptocurrency markets.
- Select the single most relevant category based on the main focus of the article.
- If an article covers multiple topics, classify it according to its primary focus.
- Always use one of the exact category names provided.

Sentiment Classification Rules:
- Positive: The news indicates favorable developments or potentially beneficial implications for the affected assets or financial markets.
- Negative: The news indicates unfavorable developments or potentially adverse implications for the affected assets or financial markets.
- Neutral: The news is primarily factual, contains balanced or mixed implications, or has no clearly identifiable positive or negative financial impact.
- Determine sentiment based on the overall financial implications of the news, not isolated positive or negative words.
- Evaluate sentiment from the perspective of the affected assets or financial markets.
- Do not automatically classify good company performance as Positive if the overall financial implications are unfavorable, or vice versa.

Risk Level Classification Rules:
- Low: The news has limited potential financial impact, involves routine developments, or is unlikely to create significant market disruption.
- Medium: The news may significantly affect a specific company, industry, asset, or market segment, but its broader market impact is likely to be limited.
- High: The news may cause substantial market volatility, systemic financial risks, or widespread disruption across financial markets.
- Assess risk based on the potential severity and scope of financial consequences, not simply whether the news is positive or negative.
- Do not assign High risk unless there is a reasonable basis for expecting significant or widespread financial consequences.

Asset Classification Rules:
- Use only standardized asset names from the supported asset list.
- Do not invent alternative asset names.
- Distinguish between stock exchanges and market indices.
- If an asset cannot be confidently mapped to the supported list, omit it.

General Rules:
- Return valid JSON only.
- Do not use Markdown code fences.
- Do not invent information that is not in the article.
- If no asset is explicitly identifiable, return an empty list.
- Distinguish between general news sentiment and potential asset impact.
- Do not claim that news caused an asset price movement unless the article explicitly establishes this.
- Separate reported facts from potential implications.
- Apply sentiment and risk classification criteria consistently across all news articles.
- Sentiment and risk level must be evaluated independently. Positive news can have High risk, and Negative news can have Low risk.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options":{
            "temperature": 0
        }
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

        # Normalize asset names
        analysis["assets"] = normalize_assets(analysis["assets"])

        # Validate the analysis
        is_valid, error = validate_news(analysis)
        if not is_valid:
            print(f"Invalid news analysis: {error}")
            return None

        return analysis

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing LLM response: {e}")
        return None


if __name__ == "__main__":

    news = """
SYDNEY, Nov 26 (Reuters) - Asian shares eked out a rare rally on Thursday after ​Wall Street fought back from deep lows, though concerns over the circular financing of AI and a new ‌round of US tariffs made for a muted mood in a holiday-thinned session.
Tech giant Nvidia (NVDA.O), opens new tab was again at the centre of the unease after S&P put its credit rating on watch for a possible downgrade citing weaknesses in the chipmaker's management and board oversight.
Make sense of global markets with the Trading Day newsletter. Sign up here.
Nvidia stock ended down 3.9% at $149.30 on Wednesday, $2.3 trillion short ​of its October peak of $243.37.
Wall Street did recoup most of its initial losses overnight as money moved into other, more ​defensive, sectors. S&P 500 futures added 0.5% early on Thursday, while Nasdaq futures firmed 0.4%.
Japan's Nikkei (.N225), opens new tab ⁠gained 1.6% and South Korea's main index (.KS11), opens new tab rose 1.9%. Japan is down 8% for the month, while South Korea has shed ​16% and Taiwan 15%.
Asia-Pacific shares outside Japan (.MISX00000PUS), opens new tab edged up 0.6%, but were still off 9% for November so far.
Bond investors are wondering ​if the broad losses in stocks coupled with doubts on the durability of AI-driven business investment might temper the hawkishness of central banks.
It was notable that Treasury yields hit multi-week lows on Wednesday even as US economic growth for the third quarter was revised up sharply to 4.3%, while the Federal ​Reserve's favoured measure of core inflation hit an uncomfortably high 3.3%.
While futures still imply a 98% chance the Fed will raise ​rates by 25 basis points in December, speculation of a half-point increase has faded.
Yields on 10-year paper were off at 4.848% , having touched an 11-week ‌trough overnight.
TARIFFS ⁠ADD TO INFLATION RISK
Still, the outlook for inflation was not helped by White House proposals for new tariffs on 14 major trading partners including China, Japan and the European Union, though crucially the new duties will not stack on top of those imposed in July.
"The latest US tariff announcement is not as alarming as it looks, and the increase in the effective tariff rate looks set to ​be a touch lower than ​we've been assuming," said Stephen ⁠Brown, chief North America economist at Capital Economics.
"The additional tariff burden is likely to be small, but will still provide a marginal boost to prices and keep core goods inflation firm into next ​year."
In currency markets, the dollar's recent strength was tempered somewhat by the drop in Treasury ​yields, with the ⁠dollar index easing a touch to 103.71 .
The safe-haven Swiss franc was a major gainer as the dollar slipped to 0.8311 francs , while the euro inched up to $1.0973 from its recent low of $1.0881.
The dollar held steady on the yen at 159.14 , with the risk of intervention keeping ⁠160.00 resistance ​intact.
In commodity markets, oil prices held firm after Yemen's Houthis said they had ​attacked Saudi Arabia's largest oil refinery, raising concerns about unexpected Middle East supply disruptions.
Brent futures rose 0.2% to $94.81 a barrel, while US crude futures added 0.6% to $84.24 ​a barrel.
Non-interest-bearing gold eased 0.4% to $3,566 an ounce , having fallen almost 8% so far this month.
Reporting by Wayne Cole; Editing by Kate Mayberry

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
