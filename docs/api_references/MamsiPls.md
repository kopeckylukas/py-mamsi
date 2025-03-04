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


---
A class that extends the MB_PLS class by extra methods convenient in Chemometrics and Metabolomics research. 
It is based on MB-PLS package: Baum et al., (2019). Multiblock PLS: Block dependent prediction modeling for Python.
This wrapper has some extra methods convenient in Chemometrics and Metabolomics research.

For a full list of methods, please refer to the MB-PLS class [documentation](https://mbpls.readthedocs.io/en/latest/index.html).


**Args**

* **n_components** (int, optional) : A number of Latent Variables (LV). Defaults to 2.
* **full_svd** (bool, optional) : Whether to use full singular value decomposition when performing the SVD method. 
    Set to False when using very large quadratic matrices (X). Defaults to False.
* **method** (str, optional) : The method used to derive the model attributes. Options are 'UNIPALS', 'NIPALS', 'SIMPLS', 
    and 'KERNEL'. Defaults to 'NIPALS'.
* **standardize** (bool, optional) : Whether to standardise the data (Unit-variance scaling). Defaults to True.
* **max_tol** (float, optional) : Maximum tolerance allowed when using the iterative NIPALS algorithm. Defaults to 1e-14.
* **nipals_convergence_norm** (int, optional) : Order of the norm that is used to calculate the difference of 
    the super-score vectors between subsequential iterations of the NIPALS algorithm. 
    Following orders are available:

    
    ord   | norm for matrices            | norm for vectors             |
    ----- | ---------------------------- | ---------------------------- |
    None  | Frobenius norm               |  2-norm                      |
    'fro' | Frobenius norm               | --                           |
    'nuc' | nuclear norm                 | --                           |
    inf   | max(sum(abs(x), axis=1))     | max(abs(x))                  |    
    -inf  | min(sum(abs(x), axis=1))     | min(abs(x))                  |
    0     | --                           | sum(x != 0)                  |
    1     | max(sum(abs(x), axis=0))     | as below                     |
    -1    | min(sum(abs(x), axis=0))     | as below                     |
    2     | 2-norm (largest sing. value) | as below                     |       
    -2    | smallest singular value      | as below                     |
    other | --                           | sum(abs(x)**ord)**(1./ord)   |

    Defaults to 2.
    
* **calc_all** (bool, optional) : Whether to calculate all internal attributes for the used method. Some methods do not need
    to calculate all attributes (i.e., scores, weights) to obtain the regression coefficients used for prediction.
    Setting this parameter to False will omit these calculations for efficiency and speed. Defaults to True.
* **sparse_data** (bool, optional) : NIPALS is the only algorithm that can handle sparse data using the method of H. Martens
    and Martens (2001) (p. 381). If this parameter is set to True, the method will be forced to NIPALS and sparse data
    is allowed. Without setting this parameter to True, sparse data will not be accepted. Defaults to False.
* **copy** (bool, optional) : Whether the deflation should be done on a copy. Not using a copy might alter the input data
    and have unforeseeable consequences. Defaults to True.


**Attributes**

* **n_components** (int) : Number of Latent Variables (LV).
* **Ts_** (array) : (X-Side) super scores [n,k]
* **T_** (list) : (X-Side) block scores [i][n,k]
* **W_** (list) : (X-Side) block weights [n][pi,k]
* **A_** (array) : (X-Side) block importances/super weights [i,k]
* **A_corrected_** (array) : (X-Side) normalized block importances (see mbpls documentation)
* **P_** (list, block) : (X-Side) loadings [i][pi,k]
* **R_** (array) : (X-Side) x_rotations R=W(PTW)-1
* **explained_var_x_** (list) : (X-Side) explained variance in X per LV [k]
* **explained_var_xblocks_** (array) : (X-Side) explained variance in each block Xi [i,k]
* **beta_** (array) : (X-Side) regression vector 𝛽 [p,q]
* **U_** (array) : (Y-Side) scoresInitialize [n,k]
* **V_** (array) : (Y-Side) loadings [q,k]
* **explained_var_y_** (list) : (Y-Side) explained variance in Y [k]



**Methods:**


### .estimate_lv
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L92)
```python
.estimate_lv(
   x, y, groups = None, max_components = 10, classification = True, metric = 'auc',
   method = 'kfold', n_splits = 5, repeats = 100, test_size = 0.2, random_state = 42,
   plateau_threshold = 0.01, increase_threshold = 0.05, get_scores = False,
   savefig = False, **kwargs
)
```

---
A method to estimate the number of latent variables (LVs)/components in the MB-PLS model.
   The method is based on cross-validation (k-fold or Monte Carlo) and combined with an outer loop with increasing number of LVs.
   LV on which the model stabilises corresponds with the optimal number of LVs.


**Args**

* **x** (array or list['array']) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : A 1-dim array of reference values, either continuous or categorical variable.
* **groups** (array, optional) : If provided, cv iterator variant with non-overlapping groups. 
    Group labels for the samples used while splitting the dataset into train/test set.
    Defaults to None.
* **max_components** (int, optional) : Maximum number of components for whic LV estimate is calculated. 
    Defaults to 10.
* **classification** (bool, optional) : Whether to perfrom calssification or regression. Defaults to True.
* **metric** (str, optional) : Metric to use to estimate the number of LVs; available options: [`AUC`, `precision`, `recall`, `f1`] for 
    categorical outcome variables and ['q2'] for continuous outcome variable. 
    Defaults to 'auc'.
* **method** (str, optional) : Corss-validation method. Available options ['kfold', 'montecarlo']. Defaults to 'kfold'.
* **n_splits** (int, optional) : Number of splits for k-fold cross-validation. Defaults to 5.
* **repeats** (int, optional) : Number of train-test split repeats from Monte Carlo. Defaults to 100.
* **test_size** (float, optional) : Test size for Monte Carlo. Defaults to 0.2.
* **random_state** (int, optional) : Generates a sequence of random splits to control MCCV. Defaults to 42.
* **plateau_threshold** (float, optional) : Maximum increase for a sequence of LVs to be considered a plateau. 
    Must be non-negative. 
    Defaults to 0.01.
* **increase_threshold** (float, optional) : Minimum increase to be considered a bend. Must be non-negative.. Defaults to 0.05.
* **get_scores** (bool, optional) : Whether to retun measured mean scores. Defaults to False.
* **savefig** (bool, optional) : Whether to save the plot as a figure. If True, argument `fname` has to be provided. Defaults to False.
* **kwargs**  : Additional keyword arguments to be passed to plt.savefig(), fname required to save .


**Raises**

* **ValueError**  : Incorrect metric for categorical outcome. Allowed values are: 'auc', 'precision', 'recall', 'specificity', 'f1', 'accuracy'.
* **ValueError**  : Incorrect metric for continuous outcome. Allowed values are: 'q2'.
* **ValueError**  : Invalid method. Available options are ['kfold', 'montecarlo'].


**Returns**

* **DataFrame**  : Measured mean scores for test and train splits for all components.


### .evaluate_class_model
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L259)
```python
.evaluate_class_model(
   x, y
)
```

---
Evaluate classification MB-PLS model using a **testing** dataset.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values - categorical variable.


**Returns**

* **array**  : Predicted y variable based on training set predictors.


### .evaluate_regression_model
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L300)
```python
.evaluate_regression_model(
   x, y
)
```

---
Evaluate regression MB-PLS model using a **testing** dataset.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values - continuous variable.


**Returns**

* **array**  : Predicted y variable based on training set predictors.


### .kfold_cv
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L342)
```python
.kfold_cv(
   x, y, groups = None, classification = True, return_train = False, n_splits = 5
)
```

---
Perform k-fold cross-validation for MB-PLS model.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values, either continuous or categorical variable.
* **groups** (array, optional) : Group labels for the samples used while splitting the dataset into train/test set.
    If provided, group k-fold is performed. 
    Defaults to None.
* **classification** (bool, optional) : Whether the outcome is a categorical variable. Defaults to True.
* **return_train** (bool, optional) : Whether to return evaluation metrics for training set. Defaults to False.
* **n_splits** (int, optional) : Number of splits for k-fold cross-validation. Defaults to 5.


**Returns**

* **DataFrame**  : Evaluation metrics for each k-fold split.
    if return_train is True, returns evaluation metrics for training set as well.


### .montecarlo_cv
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L492)
```python
.montecarlo_cv(
   x, y, groups = None, classification = True, return_train = False, test_size = 0.2,
   repeats = 10, random_state = 42
)
```

---
Evaluate MB-PLS model using Monte Carlo Cross-Validation (MCCV).


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are featuress.
* **y** (array) : 1-dim or 2-dim array of reference values - categorical variable.
* **groups** (array, optional) : Group labels for the samples used while splitting the dataset into train/test set.
    If provided, group-train-test split will be used instead of train-test split for random splits. 
    Defaults to None.
* **classification** (bool, optional) : Whether the outcome is a categorical variable. Defaults to True.
* **return_train** (bool, optional) : Whether to return evaluation metrics for training set. Defaults to False.
* **test_size** (float, optional) : Proportion of the dataset to include in the test split. Defaults to 0.2.
* **repeats** (int, optional) : Number of MCCV repeats. Defaults to 10.
* **random_state** (int, optional) : Generates a sequence of random splits to control MCCV. Defaults to 42.


**Returns**

* **DataFrame**  : Evaluation metrics for each MCCV repeat.
    if return_train is True, returns evaluation metrics for training set as well.


### .mb_vip
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L642)
```python
.mb_vip(
   plot = True, get_scores = False, savefig = False, **kwargs
)
```

---
Multi-block Variable Importance in Projection (MB-VIP) for multiblock PLS model.

Adaptation of C. Wieder et al., (2024). PathIntegrate, doi: 10.1371/journal.pcbi.1011814.


**Args**

* **plot** (bool, optional) : Whether to plot MB-VIP scores. Defaults to True.
* **get_scores** (bool, optional) : Whether to return MB-VIP scores. Defaults to False.
* **savefig** (bool, optional) : Whether to save the plot as a figure. If True, argument `fname` has to be provided. 
    Defaults to False.
* **kwargs**  : Additional keyword arguments to be passed to plt.savefig(), fname required to save .            


**Returns**

* **array**  : MB-VIP scores.


### .block_importance
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L686)
```python
.block_importance(
   block_labels = None, normalised = True, plot = True, get_scores = False,
   savefig = False, **kwargs
)
```

---
Calculate the block importance for each block in the multiblock PLS model and plot the results.


**Args**

* **block_labels** (list, optional) : List of block names. If block names are not provided or they do not match the number 
    of blocks in the model, the plot will display labels as 'Block 1', 'Block 2', ... 'Block n'. Defaults to None.
* **normalised** (bool, optional) : Whether to use normalised block importance. For more information see model attribute 
    ['A_Corrected_'](). Defaults to True.
* **plot** (bool, optional) : Whether to render plot block importance. Defaults to True.
* **get_scores** (bool, optional) : Whether to return block importance scores. Defaults to False.
* **savefig** (bool, optional) : Whether to save the plot as a figure. If True, argument `fname` has to be provided. 
    Defaults to False.
* **kwargs**  : Additional keyword arguments to be passed to plt.savefig(), fname required to save.


**Returns**

* **array**  : Block importance scores.


### .mb_vip_permtest
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L777)
```python
.mb_vip_permtest(
   x, y, n_permutations = 1000, return_scores = False
)
```

---
Calculate empirical p-values for each feature by permuting the Y outcome variable `n_permutations` times and
refitting the model. The p-values for each feature are calculated by counting the number of trials with
MB-VIP greater than or equal to the observed test statistic, and dividing this by `n_permutations`.

N.B. This method uses OpenMP to parallelise the code that relies on multi-threading exclusively. By default,
the implementations using OpenMP will use as many threads as possible, i.e. as many threads as logical cores.
This is available by default on systems with macOS and MS Windows.
Running this method on a computer clusters / High Performance Computing (HPC) system, including Imperial HPC, requires
additional Joblib parallelisation. A parallelised permutation test function can be found at function can be found in the 
[MAMSI Tutorials repository](https://github.com/kopeckylukas/py-mamsi-tutorials). If you are an Imperial colleague, 
do not hesitate to contact me for support on how to set up the configuration PBS file for this job.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values, either continuous or categorical variable.
* **n_permutations** (int, optional) : Number of permutation tests. Defaults to 1000.
* **return_scores** (bool, optional) : Whether to return MB-VIP scores for each permuted null model. Defaults to False.


**Returns**

* **array**  : Returns an array of p-values for each feature. If `return_scores` is True, then a matrix of MB-VIP scores
for each permuted null model is returned as well.

### .calculate_ci
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L840)
```python
.calculate_ci(
   data, ci_level = 0.9, dropna = True
)
```

---
Static Method

Calculates mean, margin of error, and confidence interval for each column.


**Args**

* **data** (pd.DataFrame) : The input DataFrame.
* **ci_level** (float, optional) : The confidence level (e.g., 0.90, 0.95). Defaults to 0.90.
* **dropna** (bool, optional) : Whether to drop rows containing NaNs. Defaults to True.


**Returns**

* **DataFrame**  : A DataFrame containing the calculated statistics for each column.
            If dropna = False, and a column has less than 2 valid values after
            dropping NaNs specific to that column, all the result values for that
            column will be np.nan.


### .group_train_test_split
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L883)
```python
.group_train_test_split(
   x, y, gropus = None, test_size = 0.2, random_state = 42
)
```

---
Static Method

Split the data into train and test sets based on the groups. The groups are split into train and test sets
based on the `test_size` parameter. The function returns the train and test sets for the predictors and the
response variable.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are samples, columns are features.
* **y** (array) : 1-dim or 2-dim array of reference values, either continuous or categorical variable.
* **groups** (array, optional) : Group labels for the samples used while splitting the dataset into train/test set. Defaults to None.
* **test_size** (float, optional) : Proportion of the dataset to include in the test split. Defaults to 0.2.
* **random_state** (int, optional) : Controls the shuffling applied to the data before applying the split. Defaults to 42.


**Returns**

* **tuple**  : x_train, x_test, y_train, y_test

