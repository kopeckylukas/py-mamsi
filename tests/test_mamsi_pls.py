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
def test_mamsi_pls_evaluate_regression_model(request, data_fixture):
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
    scores = model.kfold_cv(x, y, n_splits=3)
    assert not scores.empty

## test montecarlo_cv
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_montecarlo_cv(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    scores = model.montecarlo_cv(x, y, repeats=5)
    assert not scores.empty

## test mb_vip
## method does not take any user input, not needed to test other data formats
def test_mb_vip(sample_multiblock_data):
    x, y = sample_multiblock_data
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    vip_scores = model.mb_vip(plot=False, get_scores=True)
    assert vip_scores is not None

## test block_importance
## method does not take any user input, not needed to test other data formats
def test_block_importance(sample_multiblock_data):
    x, y = sample_multiblock_data
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    block_importance = model.block_importance(plot=False, get_scores=True)
    assert block_importance is not None

## test calculate_ci
def test_calculate_ci():
    data = np.random.rand(100, 5)
    df = pd.DataFrame(data)
    ci = MamsiPls.calculate_ci(df)
    assert not ci.empty

# test mb_vip_permtest
def test_mb_vip_permtest(sample_multiblock_data):
    x, y = sample_multiblock_data
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    p_vals = model.mb_vip_permtest(x, y, n_permutations=10)
    assert p_vals is not None