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
def test_mamsi_pls_initialisation():
    model = MamsiPls(n_components=3)
    assert model.n_components == 3

# Test inherited methods from the mbpls parent class
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_mamsi_pls_fit(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    assert hasattr(model, 'beta_')

@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_mamsi_pls_predict(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert len(y_pred) == len(y)

# Test MamsiPls specific methods
## test evaluate_class_model
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_mamsi_pls_evaluate_class_model(request, data_fixture):
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
def test_mamsi_pls_kfold_cv(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    scores = model.kfold_cv(x, y, n_splits=3)
    assert not scores.empty

## test montecarlo_cv
@pytest.mark.parametrize("data_fixture", ["sample_data", "sample_multiblock_data", "sample_multiblock_data_df"])
def test_mamsi_pls_montecarlo_cv(request, data_fixture):
    x, y = request.getfixturevalue(data_fixture)
    model = MamsiPls(n_components=2)
    model.fit(x, y)
    scores = model.montecarlo_cv(x, y, repeats=5)
    assert not scores.empty



