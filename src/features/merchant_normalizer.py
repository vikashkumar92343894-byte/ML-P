import re
from typing import Tuple

CLEAN_REGEX = re.compile(r"[^a-zA-Z0-9\s]")
MULTISPACE_REGEX = re.compile(r"\s+")

NOISE_WORDS = {
    "ltd", "pvt", "limited", "inc", "corp", "sro", "as", "spol",
    "pos", "terminal", "branch", "atm", "online", "web", "pay"
}

def normalize_merchant(raw_merchant: str) -> str:
    """Normalizes raw bank merchant descriptions into a consistent key."""
    if not isinstance(raw_merchant, str) or not raw_merchant.strip():
        return "unknown_merchant"
    
    # 1. Lowercase
    text = raw_merchant.lower()
    
    # 2. Extract parenthesized info or trailing IDs if present
    text = re.sub(r"\(#?\d+\)", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"acc\s*\d+", "", text)
    
    # 3. Remove punctuation except spaces
    text = CLEAN_REGEX.sub(" ", text)
    
    # 4. Remove noise tokens
    tokens = [t for t in text.split() if t not in NOISE_WORDS and len(t) > 1]
    cleaned = " ".join(tokens)
    cleaned = MULTISPACE_REGEX.sub(" ", cleaned).strip()
    
    return cleaned if cleaned else raw_merchant.strip().lower()

def extract_merchant_features(merchant_str: str) -> dict:
    norm = normalize_merchant(merchant_str)
    return {
        "norm_merchant": norm,
        "is_utility_telecom": any(w in norm for w in ["electric", "power", "gas", "telecom", "utility", "o2", "vodafone", "tmobile"]),
        "is_insurance": any(w in norm for w in ["insurance", "allianz", "generali", "pojistovna", "uniqa"]),
        "is_bank_fee": any(w in norm for w in ["fee", "bank", "maintenance"]),
        "is_atm_cash": any(w in norm for w in ["atm", "cash", "withdrawal"])
    }
