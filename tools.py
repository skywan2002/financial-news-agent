
# ==========================================
# Financial News Intelligence Agent
# ==========================================


# 1. Allowed news categories
CATEGORIES = [
    "Monetary Policy",
    "Macroeconomics",
    "Corporate",
    "Geopolitics",
    "Crypto"
]


# 2. Allowed sentiment values
SENTIMENTS = [
    "Positive",
    "Negative",
    "Neutral"
]


# 3. Allowed risk levels
RISK_LEVELS = [
    "Low",
    "Medium",
    "High"
]


# 4. Allowed financial assets
ASSETS = [
    "BTC",
    "ETH",
    "Gold",
    "S&P 500",
    "NASDAQ",
    "USD",
    "US Treasury",
    "Nvidia"
]


# 5. Required JSON fields
REQUIRED_FIELDS = [
    "category",
    "sentiment",
    "assets",
    "risk_level",
    "summary",
    "impact_rationale"
]


def normalize_assets(assets):
    """
    Normalize asset names using keyword matching.
    """

    normalized_assets = []

    # Keywords mapped to standardized asset names
    ASSET_KEYWORDS = {
        "BTC": ["btc", "bitcoin"],
        "ETH": ["eth", "ethereum"],
        "Gold": ["gold"],
        "S&P 500": ["s&p 500", "sp500", "s&p500"],
        "NASDAQ": ["nasdaq"],
        "USD": ["usd", "us dollar", "u.s. dollar"],
        "US Treasury": ["us treasury", "u.s. treasury", "treasury"],
        "Nvidia": ["nvidia"]
    }

    for asset in assets:

        if not isinstance(asset, str):
            normalized_assets.append(asset)
            continue

        asset_lower = asset.strip().casefold()

        matched_asset = None

        for standard_name, keywords in ASSET_KEYWORDS.items():

            if any(keyword in asset_lower for keyword in keywords):
                matched_asset = standard_name
                break

        if matched_asset:
            normalized_assets.append(matched_asset)
        else:
            normalized_assets.append(asset)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(normalized_assets))

def validate_news(data):
    """
    Validate the structure and values of financial news analysis.
    Return (True, None) if valid.
    Return (False, error_message) if invalid.
    """

    # 1. Check whether the input is a dictionary
    if not isinstance(data, dict):
        return False, "Input must be a dictionary."

    # 2. Check whether all required fields exist
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: {field}"

    # 3. Check whether category is valid
    if data["category"] not in CATEGORIES:
        return False, "Invalid category."

    # 4. Check whether sentiment is valid
    if data["sentiment"] not in SENTIMENTS:
        return False, "Invalid sentiment."

    # 5. Check whether risk level is valid
    if data["risk_level"] not in RISK_LEVELS:
        return False, "Invalid risk level."

    # 6. Check whether assets is a list
    if not isinstance(data["assets"], list):
        return False, "Assets must be a list."

    # 7. Check whether every asset is valid
    for asset in data["assets"]:
        if not isinstance(asset, str) or asset not in ASSETS:
            return False, f"Invalid asset: {asset}"

    # 8. Check whether summary is a non-empty string
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        return False, "Summary must be a non-empty string."

    # 9. Check whether impact_rationale is a non-empty string
    if (
        not isinstance(data["impact_rationale"], str)
        or not data["impact_rationale"].strip()
    ):
        return False, "Impact rationale must be a non-empty string."

    return True, None
