import re
from datetime import datetime
from textblob import TextBlob


KEYWORDS = [
    "refund", "cancel", "help", "problem", "broken", "down",
    "not working", "error", "failed", "issue", "charged"
]


def extract_features(text: str) -> dict:
    text = text or ""
    words = text.split()

    text_len = len(text)
    word_count = len(words)

    keyword_count = sum(1 for kw in KEYWORDS if kw in text.lower())

    exclamation_count = text.count("!")
    question_count = text.count("?")

    all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)

    uppercase_chars = sum(1 for ch in text if ch.isupper())
    total_alpha_chars = sum(1 for ch in text if ch.isalpha())
    uppercase_ratio = uppercase_chars / total_alpha_chars if total_alpha_chars > 0 else 0.0

    repeated_punct_count = len(re.findall(r"[!?]{2,}", text))

    lower_text = text.lower()
    contains_refund = 1 if "refund" in lower_text else 0
    contains_cancel = 1 if "cancel" in lower_text else 0
    contains_help = 1 if "help" in lower_text else 0
    contains_problem = 1 if "problem" in lower_text else 0

    sentiment_polarity = TextBlob(text).sentiment.polarity

    hour_of_day = datetime.now().hour

    return {
        "text_len": text_len,
        "word_count": word_count,
        "keyword_count": keyword_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "all_caps_words": all_caps_words,
        "uppercase_ratio": uppercase_ratio,
        "repeated_punct_count": repeated_punct_count,
        "contains_refund": contains_refund,
        "contains_cancel": contains_cancel,
        "contains_help": contains_help,
        "contains_problem": contains_problem,
        "sentiment_polarity": sentiment_polarity,
        "hour_of_day": hour_of_day
    }