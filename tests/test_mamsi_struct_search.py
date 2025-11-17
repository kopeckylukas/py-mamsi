# -*- coding: utf-8 -*-
#
# Authors: Lukas Kopecky <l.kopecky22@imperial.ac.uk>
#          Timothy MD Ebbels 
#          Elizabeth J Want
#
# License: BSD 3-clause

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mamsi.mamsi_struct_search import MamsiStructSearch

@pytest.fixture
def sample_lcms_data():
    data = {
        'HPOS_233.25_149.111m/z': [100, 200, 300],
        'HPOS_233.25_150.111m/z': [150, 250, 350],
        'HPOS_234.25_149.111m/z': [200, 300, 400]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_msi_data():
    data = {
        'Assay1_149.111': [100, 200, 300],
        'Assay1_150.111': [150, 250, 350],
        'Assay2_149.111': [200, 300, 400]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_lcms():
    """Load test data CSV"""
    data_path = Path(__file__).parent / "test_data" / "lcms_sample.csv"
    return pd.read_csv(data_path)

@pytest.fixture
def sample_data_msi():
    """Load test data CSV"""
    data_path = Path(__file__).parent / "test_data" / "msi_sample.csv"
    return pd.read_csv(data_path)
     

def test_load_lcms(sample_lcms_data):
    searcher = MamsiStructSearch()
    searcher.load_lcms(sample_lcms_data)
    assert searcher.intensities.equals(sample_lcms_data)
    assert searcher.feature_metadata is not None
    assert searcher.assay_links is not None

def test_load_msi(sample_data_msi):
    searcher = MamsiStructSearch()
    searcher.load_msi(sample_data_msi)
    assert searcher.intensities.equals(sample_data_msi)
    assert searcher.feature_metadata is not None
    assert searcher.assay_links is not None

def test_structural_search_msi(sample_data_msi):
    searcher = MamsiStructSearch(ppm=10)
    searcher.load_msi(sample_data_msi)
    results = searcher.get_structural_clusters(annotate=False)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
