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


@pytest.fixture(params=[
                        "msi_smpl_no_iso", 
                        "msi_smpl_no_iso_pos",
                        "msi_smpl_no_iso_neg",
                        "msi_smpl_no_adducts",
                        "msi_smpl_no_adducts_pos",
                        "msi_smpl_no_adducts_neg",
                        "msi_smpl_no_iso_no_adducts",
                        # "msi_smpl_single_assay",
                        # "msi_smpl_single_assay_no_iso",
                        # "msi_smpl_single_assay_no_adducts",
                        "msi_smpl_no_struct_pos",
                        "mis_smpl_no_struct_neg",
                        "msi_smpl_no_cross_assay",
                        "msi_smpl_all"
                        ])
def sample_data_msi_param(request):
    """Load test data CSV with different sampling strategies"""
    data_path = Path(__file__).parent / "test_data" / "msi_sample.csv"
    data = pd.read_csv(data_path)
    
    if request.param == "msi_smpl_no_iso":
        cols = np.r_[0:2, 3:7, 9:13, 14:18]
    elif request.param == "msi_smpl_no_iso_pos":
        cols = np.r_[0:13, 14:18]
    elif request.param == "msi_smpl_no_iso_neg":
        cols = np.r_[0:2, 3:7,  9:18]    
    elif request.param == "msi_smpl_no_adducts":
        cols = np.r_[1:14]
    elif request.param == "msi_smpl_no_adducts_pos":
        cols = np.r_[0:14]
    elif request.param == "msi_smpl_no_adducts_neg":
        cols = np.r_[1:18]    
    elif request.param == "msi_smpl_no_iso_no_adducts":
        cols = np.r_[0, 2:7, 9:13]
    elif request.param == "msi_smpl_single_assay":
        cols = np.r_[9:18]
    elif request.param == "msi_smpl_single_assay_no_iso":
        cols = np.r_[9:13, 14:18]
    elif request.param == "msi_smpl_single_assay_no_adducts":
        cols = np.r_[1:9]
    elif request.param == "msi_smpl_no_struct_pos":
        cols = np.r_[0:13, 17]
    elif request.param == "mis_smpl_no_struct_neg":
        cols = np.r_[0, 2, 3:7, 8:18]
    elif request.param == "msi_smpl_no_cross_assay":
        cols = np.r_[0:3, 4:18]
    else:               # "msi_smpl_all" 
        cols = np.r_[0:18]
    return data.iloc[:, cols], request.param


def test_load_lcms(sample_data_lcms):
    '''Test loading of LC-MS data import/loading'''
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
    '''Test loading of MSI data import/loading'''
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


def test_msi_iso_search(sample_data_msi_param):
    '''
    Test MSI isotopologue search functionality of structural searcher
    1. Data without isotopologues should have NaN in 'Isotopologue group' column
    2. Data with isotopologues should have at least two one not NaN value in 'Isotopologue group' column
    3. All rows with 'Isotopologue group' value should also have 'Isotopologue pattern' value
    4. Different data combinations should be tested
    5. Results should be a non-empty DataFrame
    '''
    data, param_name = sample_data_msi_param
    searcher = MamsiStructSearch(ppm=10)
    searcher.load_msi(data)
    results = searcher.get_structural_clusters(annotate=False)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    
    # Check data without isotopologues to have NaN in 'Isotopologue group'
    if param_name in [  "msi_smpl_no_iso", 
                        "msi_smpl_no_iso_no_adducts",
                        "msi_smpl_single_assay_no_iso",
                        ]:
        assert results['Isotopologue group'].isna().all(), \
            f"Expected all NaN in 'Isotopologue group' for {param_name}"

    # Check all other data combinations for isotopologue groups     
    else: 
        # Check if at least one not NaN value in results['Isotopologue group']
        assert results['Isotopologue group'].notna().any()
        # Check if at least two not NaN values in results['Isotopologue group']
        assert results['Isotopologue group'].notna().sum() >= 2, \
            f"Expected at least two not NaN  (M and M+1) in 'Isotopologue group' for {param_name}"
        # Check that all rows where Isotopologue Group has a value also have Isotopologue Pattern
        rows_with_group = results[results['Isotopologue group'].notna()]
        assert rows_with_group['Isotopologue pattern'].notna().all(), \
            "All rows with Isotopologue Group must have Isotopologue Pattern"
        
    
def test_msi_adduct_search(sample_data_msi_param):
    '''
    Test MSI adduct search functionality of structural searcher
    1. Data without adducts should have NaN in adduct-related columns
    2. Data with adducts should have at least one not NaN value in adduct-related columns
    3. Check consitency between all attributes of present adduct groups
    '''
    data, param_name = sample_data_msi_param
    
    searcher = MamsiStructSearch(ppm=10)
    searcher.load_msi(data)
    results = searcher.get_structural_clusters(annotate=False)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty

    # Data that do not contain any adducts - test that all values are NaN
    if param_name in [  "msi_smpl_no_adducts", 
                        "msi_smpl_no_iso_no_adducts",
                        "msi_smpl_single_assay_no_adducts"]:
        # Test that all values are NaN
        assert results['Adduct group'].isna().all(), \
            f"Expected all NaN in 'Adduct group' for [{param_name}]"
        assert results['Expected neutral mass'].isna().all(), \
            f"Expected all NaN in 'Expected neutral mass' for [{param_name}]"
        assert results['Observed neutral mass'].isna().all(), \
            f"Expected all NaN in 'Observed neutral mass' for [{param_name}]"
        assert results['Neutral mass |difference ppm|'].isna().all(), \
            f"Expected all NaN in 'Neutral mass |difference ppm|' for [{param_name}]"
        assert results['Adduct'].isna().all(), \
            f"Expected all NaN in 'Adduct' for [{param_name}]"
        
    # Data that countain adducts - test that at least one not NaN value exists
    else:
        # Check if at least one not NaN value exists
        assert results['Adduct group'].notna().any(), \
            f"All values are NaN in 'Adduct group' in [{param_name}]"
        assert results['Expected neutral mass'].notna().any(), \
            f"All values are NaN in 'Expected neutral mass' in [{param_name}]"
        assert results['Observed neutral mass'].notna().any(), \
            f"All values are NaN in 'Observed neutral mass' in [{param_name}]"
        assert results['Neutral mass |difference ppm|'].notna().any(), \
            f"All values are NaN in 'Neutral mass |difference ppm|' in [{param_name}]"
        assert results['Adduct'].notna().any(), \
            f"All values are NaN in 'Adduct' in [{param_name}]"
        
        # Test all rows where 'Adduct group' has a value
        rows_with_adduct_group = results[results['Adduct group'].notna()]
        assert rows_with_adduct_group['Adduct'].notna().all(), \
            f"'Adduct Group' - no corresponding 'Adduct' value in {param_name}"
        assert rows_with_adduct_group['Expected neutral mass'].notna().all(), \
            f"'Adduct Group' - no correspoinding 'Expected neutral mass' value in {param_name}"
        assert rows_with_adduct_group['Observed neutral mass'].notna().all(), \
            f"'Adduct Group' - no corresponding 'Observed neutral mass' value in {param_name}"
        assert rows_with_adduct_group['Neutral mass |difference ppm|'].notna().all(), \
            f"'Adduct Group' - no corresponding 'Neutral mass |difference ppm|' value in {param_name}" 

        # Test all rows where 'Adduct' has a value
        rows_with_adduct = results[results['Adduct'].notna()]
        assert rows_with_adduct['Expected neutral mass'].notna().all(), \
            f"'Adduct' - no corresponding 'Expected neutral mass' value in [{param_name}]"
        assert rows_with_adduct['Observed neutral mass'].notna().all(), \
            f"'Adduct' - no corresponding 'Observed neutral mass' value in [{param_name}]"
        assert rows_with_adduct['Neutral mass |difference ppm|'].notna().all(), \
            f"'Adduct' - no corresponding 'Neutral mass |difference ppm|' value in [{param_name}]"
        assert rows_with_adduct['Adduct group'].notna().all(), \
            f"'Adduct' - no corresponding 'Adduct group' value in [{param_name}]"


@pytest.mark.skip(reason="Not implemented yet")
def test_msi_struct_group_search(sample_data_msi_param):
    data, param_name = sample_data_msi_param
    
    # Skip specific parameter combinations
    if param_name in [  "msi_smpl_no_iso", 
                        "msi_smpl_no_iso_no_adducts",
                        "msi_smpl_single_assay_no_iso"]:
        pytest.skip(f"Skipping {param_name} for this test")
    
    searcher = MamsiStructSearch(ppm=10)
    searcher.load_msi(data)
    results = searcher.get_structural_clusters(annotate=False)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty

    
@pytest.mark.skip(reason="Not implemented yet")
def test_msi_cross_assay_search(sample_data_msi_param):
    data, param_name = sample_data_msi_param
    
    # Skip specific parameter combinations
    if param_name in [  "msi_smpl_no_iso", 
                        "msi_smpl_no_iso_no_adducts",
                        "msi_smpl_single_assay_no_iso"]:
        pytest.skip(f"Skipping {param_name} for this test")
    
    searcher = MamsiStructSearch(ppm=10)
    searcher.load_msi(data)
    results = searcher.get_structural_clusters(annotate=False)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty

# @pytest.mark.xfail(reason="Should fail but doesn't")
# def test_unexpectedly_works():
#     assert True  # This passes, but we expected it to fail!

# # With strict=True, XPASS becomes a failure
# @pytest.mark.xfail(strict=True)
# def test_must_fail():
#     assert True  # This will cause the test suite to fail