import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import parse_age

def test_parse_age_normal():
    assert parse_age('[70-80)') == 75.0
    assert parse_age('[0-10)') == 5.0
    assert parse_age('[40-50)') == 45.0

def test_parse_age_nan():
    assert np.isnan(parse_age(np.nan))
    assert np.isnan(parse_age('invalid'))
