import pytest
import pandas as pd
import numpy as np
from src.labeling.rules import label_recurring_rules, detect_price_increases, detect_duplicates, get_cadence_frequency

def test_label_recurring_rules():
    df = pd.DataFrame({
        "mean_gap": [30.5, 7.0, 365.0, 14.0],
        "gap_cv": [0.05, 0.10, 0.02, 0.80],
        "amt_cv": [0.02, 0.01, 0.05, 0.40],
        "n_txn": [12, 10, 3, 2]
    })
    
    recurring = label_recurring_rules(df)
    assert recurring.iloc[0] == True   # Monthly regular
    assert recurring.iloc[1] == True   # Weekly regular
    assert recurring.iloc[2] == True   # Yearly regular
    assert recurring.iloc[3] == False  # Irregular / small history

def test_detect_price_increases():
    df = pd.DataFrame({
        "is_recurring": [True, True, False],
        "pct_change": [0.10, 0.01, 0.15],
        "n_txn": [5, 5, 5]
    })
    hikes = detect_price_increases(df, min_pct=0.03)
    assert hikes.iloc[0] == True   # Recurring with 10% increase
    assert hikes.iloc[1] == False  # Recurring with 1% increase
    assert hikes.iloc[2] == False  # Non-recurring

def test_detect_duplicates():
    df = pd.DataFrame({
        "hrs_since_same": [2.5, 36.0, np.nan]
    })
    dups = detect_duplicates(df, max_hours=24.0)
    assert dups.iloc[0] == True
    assert dups.iloc[1] == False
    assert dups.iloc[2] == False

def test_cadence_frequency():
    assert get_cadence_frequency(30.0) == "Monthly"
    assert get_cadence_frequency(7.0) == "Weekly"
    assert get_cadence_frequency(365.0) == "Yearly"
    assert get_cadence_frequency(15.0) == "Ad-hoc"
