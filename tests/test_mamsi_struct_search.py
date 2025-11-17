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
def sample_data_lcms():
    """Load test data CSV"""
    data_path = Path(__file__).parent / "test_data" / "lcms_sample.csv"
    return pd.read_csv(data_path)

@pytest.fixture
def sample_data_msi():
    """Load test data CSV"""
    data_path = Path(__file__).parent / "test_data" / "msi_sample.csv"
    return pd.read_csv(data_path)
     

def test_load_lcms(sample_data_lcms):
    searcher = MamsiStructSearch()
    searcher.load_lcms(sample_data_lcms)
    assert searcher.intensities.equals(sample_data_lcms)
    assert searcher.feature_metadata is not None
    assert searcher.assay_links is not None
    assert sample_data_lcms.shape[1] == searcher.intensities.shape[1]
    assert len(searcher.feature_metadata) == searcher.intensities.shape[1]

    # Check column data types
    expected_dtypes = {
        'Feature': 'object',
        'Assay': 'object',
        'RT': np.float64,
        'm/z': np.float64
    }
    assert searcher.feature_metadata[list(expected_dtypes.keys())].dtypes.to_dict() == expected_dtypes



def test_load_msi(sample_data_msi):
    searcher = MamsiStructSearch()
    searcher.load_msi(sample_data_msi)
    assert searcher.intensities.equals(sample_data_msi)
    assert searcher.feature_metadata is not None
    assert searcher.assay_links is not None
    assert len(searcher.feature_metadata) == searcher.intensities.shape[1]
    assert sample_data_msi.shape[1] == searcher.intensities.shape[1]

    # Check column data types
    expected_dtypes = {
        'Feature': 'object',
        'Assay': 'object',
        'RT': np.float64,
        'm/z': np.float64
    }
    assert searcher.feature_metadata[list(expected_dtypes.keys())].dtypes.to_dict() == expected_dtypes



# def test_structural_search_msi(sample_data_msi):
#     searcher = MamsiStructSearch(ppm=10)
#     searcher.load_msi(sample_data_msi)
#     results = searcher.get_structural_clusters(annotate=False)
#     assert isinstance(results, pd.DataFrame)
#     assert not results.empty
