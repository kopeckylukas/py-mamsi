#


## MamsiPls
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L22)
```python 
MamsiPls(
   n_components = 2, full_svd = False, method = 'NIPALS', standardize = True,
   max_tol = 1e-14, nipals_convergence_norm = 2, calc_all = True, sparse_data = False,
   copy = True
)
```




**Methods:**


### .estimate_lv
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L70)
```python
.estimate_lv(
   x, y, n_components = 10, no_fold = 5, y_continuous = False, metric = 'auc',
   plateau_threshold = 0.01, increase_threshold = 0.05, get_scores = False
)
```

---
Method to estimate the number of latent variables (components) for MAMSI MB-PLS model.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values, either continuous or categorical variable.
* **n_components** (int, optional) : Number of components / latent variables. Defaults to 10.
* **no_fold** (int, optional) : Number of folds for k-fold cross-validation. Defaults to 5.
* **y_continuous** (bool, optional) : Whether the outcome is a continuous variable. Defaults to False.
* **metric** (str, optional) : Metric to use to estimate the number of LVs; available options: ['AUC', 'q2', 'precision', 'recall', 'f1']. Defaults to 'AUC'.
* **plateau_threshold** (float, optional) : Maximum increase for a sequence of LVs to be considered a plateau. Must be non-negative. Defaults to 0.01.
* **increase_threshold** (float, optional) : Minimum increase to be considered a bend. Must be non-negative. Defaults to 0.05.
* **get_scores** (bool, optional) : Whether to return measured scores as a dataframe. Defaults to False.


**Returns**

* **DataFrame**  : Measured scores as a Pandas dataframe.


### .evaluate_class_model
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L273)
```python
.evaluate_class_model(
   x, y
)
```

---
Evaluate classfication MB-PLS model using a **testing** dataset.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values - categorical variable.


**Returns**

* **array**  : Predicted y variable based on training set predictors.


### .evaluate_regression_model
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L314)
```python
.evaluate_regression_model(
   x, y
)
```

---
Evaulate regression MB-PLS model using a **testing** dataset.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values - continuous variable.


**Returns**

* **array**  : Predicted y variable based on training set predictors.


### .mb_vip
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L356)
```python
.mb_vip(
   plot = False
)
```

---
Multi-block Variable Importance in Projection (MB-VIP) for multiblock PLS model.

Adaptation of C. Wieder et al., (2024). PathIntegrate, doi: 10.1371/journal.pcbi.1011814.


**Args**

* **plot** (bool, optional) : Whether to plot MB-VIP scores. Defaults to False.


**Returns**

* **array**  : MB-VIP scores.


### .mb_vip_permtest
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L391)
```python
.mb_vip_permtest(
   x, y, n_permutations = 1000, return_scores = False
)
```

---
Calculate empirical p-values for each feature by permuting the Y outcome variable `n_permutations` times and
refitting the model. The p-values for each feature are then calculated by counting the number of trials with
MB-VIP greater than or equal to the observed test statistic, and dividing this by `n_permutations`.

N.B. This method uses OpenMP to parallelise the code, relying on multi-threading exclusively. By default,
the implementations using OpenMP will use as many threads as possible, i.e. as many threads as logical cores.
This is available by default on systems with macOS and MS Windows.
Running this method on a High Performance Computing (HPC) system, including Imperial College London HPC, requires
additional Joblib parallelisation. A parallelised permtest function can be found in the ./Extras directory
as `parallel_mb_vip_permtest.py`. If you are an Imperial colleague, do not hesitate to contact me for support on how
to set up a PBS file.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values, either continuous or categorical variable.
* **n_permutations** (int, optional) : Number of permutation tests. Defaults to 1000.
* **return_scores** (bool, optional) : Whether to return MB-VIP scores for each permuted null model. Defaults to False.


**Returns**

* **array**  : Returns an array of p-values for each feature. If `return_scores` is True, then a matrix of MB-VIP scores
for each permuted null model is returned as well.

### ._find_plateau
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L454)
```python
._find_plateau(
   scores, range_threshold = 0.01, consecutive_elements = 3
)
```

---
Function to assist in finding a plateau in a sequence of LVs.


**Args**

* **scores** (list[float]) : List of scores.
* **range_threshold** (float, optional) : Maximum increase for a sequence of LVs to be considered a plateau. Defaults to 0.01.
* **consecutive_elements** (int, optional) : Number of elements that need to be in a plateau. Defaults to 3.


**Returns**

* **tuple**  : Beginning and end indices of the plateau.

