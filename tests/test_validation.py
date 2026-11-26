
from tools import validate_news


# Test data: valid news analysis
valid_news = {
    "category": "Corporate",
    "sentiment": "Positive",
    "assets": ["Nvidia"],
    "risk_level": "Medium",
    "summary": "Nvidia reported stronger-than-expected revenue.",
    "impact_rationale": "Strong revenue may support investor confidence."
}


def test_valid_news():
    is_valid, error = validate_news(valid_news)

    assert is_valid is True
    assert error is None


def test_invalid_category():
    news = valid_news.copy()
    news["category"] = "Technology"

    is_valid, error = validate_news(news)

    assert is_valid is False
    assert error == "Invalid category."


def test_missing_field():
    news = valid_news.copy()
    del news["summary"]

    is_valid, error = validate_news(news)

    assert is_valid is False
    assert error == "Missing required field: summary"


def test_invalid_assets_type():
    news = valid_news.copy()
    news["assets"] = "Nvidia"

    is_valid, error = validate_news(news)

    assert is_valid is False
    assert error == "Assets must be a list."

def test_invalid_sentiment():
    news = valid_news.copy()
    news["sentiment"] = "Very Positive"

    is_valid, error = validate_news(news)

    assert is_valid is False
    assert error == "Invalid sentiment."