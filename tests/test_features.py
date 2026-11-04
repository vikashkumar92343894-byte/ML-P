import pytest
import pandas as pd
import numpy as np
from src.features.merchant_normalizer import normalize_merchant, extract_merchant_features
from src.features.build_features import build_features

def test_normalize_merchant():
    assert normalize_merchant("NETFLIX.COM 1234") == "netflix com 1234"
    assert normalize_merchant("Spotify Ltd.") == "spotify"
    assert normalize_merchant("Prague Power & Electric (AB) (#12345)") == "prague power electric"
    assert normalize_merchant("") == "unknown_merchant"

def test_extract_merchant_features():
    feats = extract_merchant_features("Allianz Insurance Company")
    assert feats["is_insurance"] is True
    assert feats["is_atm_cash"] is False

def test_build_features_computation():
    dates = pd.date_range("1996-01-01", periods=4, freq="30D")
    df = pd.DataFrame({
        "customer_id": [1, 1, 1, 1],
        "merchant": ["Netflix", "Netflix", "Netflix", "Netflix"],
        "date": dates,
        "amount": [199.0, 199.0, 199.0, 199.0]
    })
    
    feats = build_features(df)
    assert len(feats) == 4
    assert feats["n_txn"].iloc[0] == 4
    assert feats["amt_cv"].iloc[0] == 0.0
    assert abs(feats["mean_gap"].iloc[0] - 30.0) < 1.0
    assert feats["gap_cv"].iloc[0] == 0.0
    assert feats["is_approx_monthly"].iloc[0] == 1
