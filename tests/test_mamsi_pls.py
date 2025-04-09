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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mamsi.mamsi_pls import MamsiPls


# Fixtures
## To work with single block data
@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return x, y

## To work with multi-block data
@pytest.fixture
def sample_multiblock_data():
    np.random.seed(42)
    x1 = np.random.rand(100, 10)
    x2 = np.random.rand(100, 15)
    y = np.random.randint(0, 2, 100)
    return [x1, x2], y

## To work with multi-block data in pandas DataFrame format
@pytest.fixture
def sample_multiblock_data_df():
    np.random.seed(42)
    x1 = pd.DataFrame(np.random.rand(100, 10))
    x2 = pd.DataFrame(np.random.rand(100, 15))
    y = np.random.randint(0, 2, 100)
    return [x1, x2], y

# Tests
## Initialisation
def test_initialisation():
    model = MamsiPls(n_components=3)
    assert model.n_components == 3

# Test inherited methods from the mbpls parent class
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_fit(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    assert hasattr(model, 'beta_')

@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_predict(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert len(y_pred) == len(y)

# Test MamsiPls specific methods
## test evaluate_class_model
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_evaluate_class_model(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    y_pred = model.evaluate_class_model(x, y)
    assert len(y_pred) == len(y)

## test evaluate_regression_model
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_evaluate_regression_model(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    y = np.random.rand(100)  # Continuous target for regression
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    y_pred = model.evaluate_regression_model(x, y)
    assert len(y_pred) == len(y)

## test kfold_cv
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_kfold_cv(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    scores = model.kfold_cv(x, y, n_splits=3, n_jobs=1)
    assert not scores.empty

## test montecarlo_cv
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_montecarlo_cv(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    scores = model.montecarlo_cv(x, y, repeats=5, n_jobs=1)
    assert not scores.empty

# test montecarlo_cv with different number of repeats
@pytest.mark.parametrize("repeats", [1, 5, 10])
def test_montecarlo_cv_repeats(sample_multiblock_data, repeats):
    x, y = sample_multiblock_data
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    scores = model.montecarlo_cv(x, y, repeats=repeats, n_jobs=1)
    assert not scores.empty

# test montecarlo_cv with different data sizes
@pytest.mark.parametrize("data_size", [(50, 5), (200, 20)])
def test_montecarlo_cv_data_sizes(data_size):
    np.random.seed(42)
    x1 = np.random.rand(data_size[0], data_size[1])
    x2 = np.random.rand(data_size[0], data_size[1] + 5)
    y = np.random.randint(0, 2, data_size[0])
    x = [x1, x2]
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    scores = model.montecarlo_cv(x, y, repeats=5, n_jobs=1)
    assert not scores.empty

# test montecarlo_cv with empty data blocks
def test_montecarlo_cv_empty_data():
    x = [np.empty((0, 10)), np.empty((0, 15))]
    y = np.empty((0,))
    model = MamsiPls(n_components=2)
    with pytest.raises(ValueError):
        model.fit(x, y)
        model.montecarlo_cv(x, y, repeats=5, n_jobs=1)

# test montecarlo_cv with different random states
@pytest.mark.parametrize("random_state", [0, 42, 100])
def test_montecarlo_cv_random_state(sample_multiblock_data, random_state):
    x, y = sample_multiblock_data
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    scores_1 = model.montecarlo_cv(x, y, repeats=5, random_state=random_state, n_jobs=1)
    scores_2 = model.montecarlo_cv(x, y, repeats=5, random_state=random_state, n_jobs=1)
    assert scores_1.equals(scores_2)

# test mb_vip_permtest with different number of permutations
@pytest.mark.parametrize("n_permutations", [10, 50, 100])
def test_mb_vip_permtest_permutations(sample_multiblock_data, n_permutations):
    x, y = sample_multiblock_data
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    p_vals = model.mb_vip_permtest(x, y, n_permutations=n_permutations, n_jobs=1)
    assert p_vals is not None
    assert len(p_vals) == sum([block.shape[1] for block in x])

# test mb_vip_permtest with return_scores=True
def test_mb_vip_permtest_return_scores(sample_multiblock_data):
    x, y = sample_multiblock_data
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    p_vals, vip_scores = model.mb_vip_permtest(x, y, n_permutations=100, return_scores=True, n_jobs=1)
    assert p_vals is not None
    assert vip_scores is not None
    assert len(p_vals) == sum([block.shape[1] for block in x])
    assert vip_scores.shape[1] == 100

# test mb_vip_permtest with different data formats
@pytest.mark.parametrize("data_fixture", ["sample_multiblock_data", "sample_multiblock_data_df"])
def test_mb_vip_permtest_data_formats(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    p_vals = model.mb_vip_permtest(x, y, n_permutations=10, n_jobs=1)
    assert p_vals is not None
    assert len(p_vals) == sum([block.shape[1] for block in x])

# test mb_vip_permtest with different n_components
@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_mb_vip_permtest_n_components(sample_multiblock_data, n_components):
    x, y = sample_multiblock_data
    model = MamsiPls(n_components=n_components)
    model.fit(x, y)
    p_vals = model.mb_vip_permtest(x, y, n_permutations=10, n_jobs=1)
    assert p_vals is not None
    assert len(p_vals) == sum([block.shape[1] for block in x])