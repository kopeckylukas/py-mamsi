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
    else: # "msi_smpl_all" 
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
    '''Test MSI isotopologue search functionality of structural searcher
        Data without isotopologues should have NaN in 'Isotopologue group' column
        Data with isotopologues should have at least two one not NaN value in 'Isotopologue group' column
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
        


@pytest.mark.skip(reason="Not implemented yet")
def test_msi_adduct_search(sample_data_msi_param):
    data, param_name = sample_data_msi_param
    
    searcher = MamsiStructSearch(ppm=10)
    searcher.load_msi(data)
    results = searcher.get_structural_clusters(annotate=False)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert False

        # # Skip specific parameter combinations
    # if param_name in [  "msi_smpl_no_iso", 
    #                     "msi_smpl_no_iso_no_adducts",
    #                     "msi_smpl_single_assay_no_iso"]:
    #     pytest.skip(f"Skipping {param_name} for this test")



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